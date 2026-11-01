"""forex_bot.py'nin kurallarını 2023-01-01 – 2026-09-10 geçmişinde portföy olarak çalıştırır (ileri testten önce ölçüm).

Bot gibi: 28 parite, 5M adım; ML 4 saatlik yön modeli (2023 öncesiyle eğitildi), 1,5 × 15M ATR stop,
Karar giriş şartı (hedefin stoptan önce gelme oranı ≥ %60, maliyet hedefin yarısından az), 500 $ sanal bakiye,
%2,5 risk, hedef 0,3 R, en fazla 3 pozisyon, para biriminde aynı yönde en fazla 1, teminat ≤ bakiyenin yarısı
(1:200), parite başına saatlik sinyalde tek giriş, New York gün sonu ve yüksek etkili haber (60 dk önce – 15 dk sonra)
filtresi. Yönetim: hedefte stop giriş + maliyete, sonra en iyi fiyatın 1 ATR gerisinden takip; stopun yarısında çıkış;
saat başında ML yönü dönerse çıkış; takip yoksa 24 saatte çıkış; 4 saatte hedefe gelmemiş zararsız pozisyonu
yeni adaya bırakma (takas).

Varyantlar her kuralın katkısını ayırır; kurallar ve varyantlar sonuçtan önce protocol.json'a yazıldı.
Fiyatlar 5M mum açılış/yüksek/düşük; aynı mumda hem stop hem hedef/ilerleme varsa önce kötü olan sayılır.
Maliyet: girişin New York saatindeki ölçülmüş spread + 0,7 pip komisyon. Anlık spread sıçraması ve kayma
modellenmedi (stres varyantı +0,5 pip).

    python -X utf8 research_bot_backtest.py            # tüm varyantlar, research/bot_backtest/
    python -X utf8 research_bot_backtest.py --pilot    # 2023 ilk çeyrek, yalnız bot (süre ölçümü)
    python -X utf8 research_bot_backtest.py --basket-trail --fixed-risk  # canlı botun sepet takibi ızgarası
    python -X utf8 research_bot_backtest.py --support --fixed-risk       # 15M/5M/oynaklık desteği
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from forex_config import get_pip_size
from forex_costs import ECN_COMMISSION_PIPS, load_spread_profile, measured_spread_pips
from forex_decision import MIN_HIT, MAX_COST_OF_TARGET
from forex_analysis import bias_score_frame
from forex_indicators import add_indicators
from forex_ml_live import load_research_model
from forex_ml_tournament import build_features
from forex_trade_plan import load_plan_stats
from research_consensus import PAIRS, ml_probabilities, read_archive

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "research" / "bot_backtest"
CACHE = OUT / "cache"
START, END = pd.Timestamp("2023-01-01", tz="UTC"), pd.Timestamp("2026-09-10", tz="UTC")
NEWS_DIRS = [ROOT / "research/news_drift/raw_forexfactory", ROOT / "research/calendar_sources/raw/forexfactory"]
USD_PAIRS = {"EUR": ("EURUSD", 1), "GBP": ("GBPUSD", 1), "AUD": ("AUDUSD", 1), "NZD": ("NZDUSD", 1),
             "CAD": ("USDCAD", -1), "CHF": ("USDCHF", -1), "JPY": ("USDJPY", -1)}
WORKERS = 7

BALANCE, RISK_PCT, TARGET_R = 500.0, 2.5, 0.3
MAX_POSITIONS, MAX_CURRENCY, LEVERAGE, MAX_MARGIN_SHARE = 3, 1, 200.0, 0.5
STOP_ATR, TRAIL_ATR, EARLY_EXIT_SHARE = 1.5, 1.0, 0.5
HOLD_HOURS, STALE_HOURS = 24, 4
CONTRACT, MIN_LOT = 100_000, 0.01


@dataclass(frozen=True)
class Variant:
    name: str
    trailing: bool = True       # hedefte stop başabaşa, sonra takip; False: hedefte kapat (sabit TP)
    half_stop: bool = True      # stopun yarısında çık; False: tam stop
    ml_exit: bool = True        # saat başında ML yönü dönerse çık
    swap: bool = True           # takılmış zararsız pozisyonu yeni adaya bırak
    direction: int = 1          # 1: ML yönü, -1: tersi (kontrol)
    extra_cost: float = 0.0     # stres: işlem başına ek kayma (pip)
    use_target: bool = True     # False: işlem başına hedef/takip yok (yalnız stop ve sepet)
    basket_pct: float = 0.0     # >0: açık pozisyonların toplam net kârı bakiyenin bu kadarı olunca hepsini kapat
    basket_optimistic: bool = False  # sepeti mum içinde en iyi fiyatlarla da kontrol et (üst sınır)
    basket_trail: bool = False  # basket_pct'e gelince hemen değil, tepeden geri verince / dönünce kapat
    basket_giveback: float = 0.3  # basket_trail: tepenin bu kadarı geri verilince kapat
    need_15m: bool = False      # destek: son kapanmış 15M teknik skoru giriş yönünde >= SUPPORT_SCORE
    need_5m: bool = False       # destek: aynısı 5M'de
    vol: int = 0                # destek: 1 yalnız oynaklık yüksek, -1 yalnız düşük (4 saatlik model)
    floor_atr: float | None = None  # takip tabanı: None giriş + maliyet; x giriş − x ATR; inf taban yok
    min_conf: float = 0.0       # ML olasılığı yön tarafında en az bu (ör. 0,55); 0 şart yok
    cooldown_hours: float = 0.0  # zararla kapanan paritede bu kadar saat giriş yok
    exit_conf: float = 0.5      # ML dönüş çıkışı: ters yönde en az bu güven
    exit_confirm: int = 1       # ML dönüş çıkışı: üst üste bu kadar saat başı ters
    max_cost_share: float = 1.0  # isteğe bağlı maliyet / (stop + maliyet) giriş sınırı


VARIANTS = [
    Variant("Bot (şu anki kurallar)"),
    Variant("Sabit hedef (Karar kartı planı)", trailing=False, half_stop=False, ml_exit=False, swap=False),
    Variant("Bot, yarım stop yok", half_stop=False),
    Variant("Bot, ML dönüş çıkışı yok", ml_exit=False),
    Variant("Bot, takas yok", swap=False),
    Variant("Bot, ML'nin ters yönü (kontrol)", direction=-1),
    Variant("Bot, +0,5 pip kayma (stres)", extra_cost=0.5),
]

# protocol_basket.json: kullanıcının "3 işlemden toplam 15 $ olunca çık" fikri (500 $'ın %3'ü)
_STOP_ONLY = dict(use_target=False, half_stop=False, ml_exit=False, swap=False)
BASKET_VARIANTS = [
    Variant("B1 toplam +15 $ + botun kuralları", basket_pct=0.03),
    Variant("B2 toplam +15 $ + yalnız tam stop", basket_pct=0.03, **_STOP_ONLY),
    Variant("B3 toplam +15 $ + yarım stop", basket_pct=0.03, **{**_STOP_ONLY, "half_stop": True}),
    Variant("B4 B2, ML'nin ters yönü (kontrol)", basket_pct=0.03, direction=-1, **_STOP_ONLY),
]

# protocol_basket_trail.json: canlı botun BasketWatch kuralı (toplam tepe yaptıktan sonra geri verince/dönünce çık)
TRAIL_STALL_STEPS = 2  # 10 dk yeni tepe yok
TRAIL_BASES = {"bot": {}, "stop": _STOP_ONLY}
TRAIL_GRID = [(arm, give) for arm in (0.02, 0.03, 0.05) for give in (0.2, 0.3, 0.5)]
TRAIL_VARIANTS = [
    Variant(f"T {base} arm {arm:.0%} geri {give:.0%}{' ters' if way == -1 else ''}", basket_pct=arm,
            basket_trail=True, basket_giveback=give, direction=way, **rules)
    for base, rules in TRAIL_BASES.items() for arm, give in TRAIL_GRID for way in (1, -1)
]
CHOOSE_END = pd.Timestamp("2025-01-01", tz="UTC")

# protocol_support.json: ML girişine 15M/5M skoru ve 4 saatlik oynaklık desteği
SUPPORT_SCORE, VOL_THRESHOLD = 25.0, 0.5
_LIVE_BASKET = dict(basket_pct=0.03, basket_trail=True, basket_giveback=0.3)
_SUPPORTS = [("S0 destek yok", {}), ("S1 ML + 15M", {"need_15m": True}),
             ("S2 ML + 15M + 5M", {"need_15m": True, "need_5m": True}),
             ("S3 ML + oynaklık yüksek", {"vol": 1}), ("S4 ML + oynaklık düşük", {"vol": -1}),
             ("S5 ML + 15M + 5M + oynaklık yüksek", {"need_15m": True, "need_5m": True, "vol": 1}),
             ("S6 ML + 15M + 5M + oynaklık düşük", {"need_15m": True, "need_5m": True, "vol": -1})]
FLOOR_VARIANTS = [Variant(name, **_LIVE_BASKET, floor_atr=x) for name, x in (
    ("F0 taban giriş + maliyet", None), ("F1 taban giriş − 0,25 ATR", 0.25), ("F2 taban giriş − 0,5 ATR", 0.5),
    ("F3 taban giriş − 1 ATR", 1.0), ("F4 taban yok", math.inf))]
_LIVE_NOW = dict(**_LIVE_BASKET, floor_atr=0.5)
CONF_VARIANTS = [Variant(name, direction=way, **_LIVE_NOW, **rules) for name, rules, way in (
    ("C0 taban", {}, 1), ("C1 güven ≥ 0,52", {"min_conf": 0.52}, 1), ("C2 güven ≥ 0,55", {"min_conf": 0.55}, 1),
    ("C3 güven ≥ 0,60", {"min_conf": 0.60}, 1), ("D1 zarardan sonra 4 saat bekle", {"cooldown_hours": 4}, 1),
    ("CD güven ≥ 0,55 + 4 saat bekle", {"min_conf": 0.55, "cooldown_hours": 4}, 1),
    ("C2 ters", {"min_conf": 0.55}, -1), ("D1 ters", {"cooldown_hours": 4}, -1))]
SUPPORT_VARIANTS = [Variant(name + (" ters" if way == -1 else ""), direction=way, **_LIVE_BASKET, **rules)
                    for name, rules in _SUPPORTS for way in (1, -1)]


# ----------------------------------------------------------------------------- veri

def high_impact_events() -> dict[str, np.ndarray]:
    """Para birimi başına yüksek etkili haber zamanları (UTC, ns), ForexFactory haftalık arşivinden."""
    times: dict[str, set] = {}
    for folder in NEWS_DIRS:
        for path in sorted(folder.glob("*.json")):
            for day in json.loads(path.read_text(encoding="utf-8")):
                for event in day["events"]:
                    if event.get("impactName") == "high" and event.get("dateline"):
                        times.setdefault(event["currency"], set()).add(int(event["dateline"]) * 10**9)
    return {c: np.array(sorted(v), dtype=np.int64) for c, v in times.items()}


def pair_frame(pair: str) -> pd.DataFrame:
    """5M açılış/yüksek/düşük, son kapanmış 15M ATR ve son kapanmış saatlik mumun ML olasılığı."""
    cached = CACHE / f"{pair}.pkl"
    if cached.exists():
        return pd.read_pickle(cached)
    bars5 = read_archive("historical_5m", pair, START)
    bars5 = bars5[bars5.index < END]
    bars15 = read_archive("historical_15m", pair, START - pd.Timedelta(days=30))
    atr = add_indicators(bars15[["Open", "High", "Low", "Close"]])["ATR14"]
    atr.index = atr.index + pd.Timedelta(minutes=15)  # kapanış zamanı
    hourly = read_archive("historical_1h_from_15m", pair, pd.Timestamp("2000-01-01", tz="UTC"))
    probs = ml_probabilities(pair, hourly)
    probs.index = probs.index + pd.Timedelta(hours=1)  # kapanış zamanı: uygulama son kapanmış saati puanlar
    out = bars5[["Open", "High", "Low"]].copy()
    out["ATR"] = atr.reindex(out.index, method="ffill")
    out["Prob"] = probs.reindex(out.index, method="ffill")
    CACHE.mkdir(parents=True, exist_ok=True)
    out.to_pickle(cached)
    return out


def support_frame(pair: str) -> pd.DataFrame:
    """Son kapanmış 15M ve 5M teknik skoru ve 4 saatlik oynaklık olasılığı, 5M açılışlarında."""
    cached = CACHE / f"{pair}_support.pkl"
    if cached.exists():
        return pd.read_pickle(cached)
    index = pair_frame(pair).index
    out = pd.DataFrame(index=index)
    for column, folder, minutes, warmup in (("S15", "historical_15m", 15, 30), ("S5", "historical_5m", 5, 10)):
        bars = read_archive(folder, pair, START - pd.Timedelta(days=warmup))
        score = bias_score_frame(bars)["score"]
        score.index = score.index + pd.Timedelta(minutes=minutes)  # kapanış zamanı
        out[column] = score.reindex(index, method="ffill")
    hourly = read_archive("historical_1h_from_15m", pair, pd.Timestamp("2000-01-01", tz="UTC"))
    bundle = load_research_model(pair, "high_volatility_4h")
    features, _ = build_features(hourly[["Open", "High", "Low", "Close"]], pair)
    rows = features[bundle["columns"]]
    rows = rows[rows.index >= START - pd.Timedelta(days=7)]
    usable = rows.notna().all(axis=1)
    vol = pd.Series(np.nan, index=rows.index)
    vol[usable] = bundle["model"].predict_proba(rows[usable])[:, 1]
    vol.index = vol.index + pd.Timedelta(hours=1)
    out["Vol"] = vol.reindex(index, method="ffill")
    out.to_pickle(cached)
    return out


def load_market(parallel: bool = True) -> dict:
    if parallel:
        with Pool(WORKERS) as pool:
            frames = dict(zip(PAIRS, pool.map(pair_frame, PAIRS)))
            supports = dict(zip(PAIRS, pool.map(support_frame, PAIRS)))
    else:  # önbellekten; havuz işçisi içinde yeni havuz açılamaz
        frames = {pair: pair_frame(pair) for pair in PAIRS}
        supports = {pair: support_frame(pair) for pair in PAIRS}
    frames = {pair: frames[pair].join(supports[pair]) for pair in PAIRS}
    index = sorted(set().union(*(f.index for f in frames.values())))
    index = pd.DatetimeIndex(index)
    arrays = {k: np.full((len(index), len(PAIRS)), np.nan)
              for k in ("Open", "High", "Low", "ATR", "Prob", "S15", "S5", "Vol")}
    for j, pair in enumerate(PAIRS):
        aligned = frames[pair].reindex(index)
        for k in arrays:
            arrays[k][:, j] = aligned[k].to_numpy()
    # para birimlerinin USD karşılığı (pip değeri ve teminat için), açılışlar ileri doldurulur
    usd = {"USD": np.ones(len(index))}
    for currency, (pair, way) in USD_PAIRS.items():
        price = pd.Series(arrays["Open"][:, PAIRS.index(pair)]).ffill().bfill().to_numpy()
        usd[currency] = price if way == 1 else 1 / price
    ny = index.tz_convert("America/New_York")
    profile = load_spread_profile()
    spread = np.array([[measured_spread_pips(profile, pair, h) or 1.5 for pair in PAIRS] for h in range(24)])
    return {"index": index, "ns": index.asi8, "ny_minutes": (ny.hour * 60 + ny.minute).to_numpy(),
            "ny_hour": ny.hour.to_numpy(), "ny_weekday": ny.weekday.to_numpy(), "usd": usd,
            "spread": spread, **arrays}


# ----------------------------------------------------------------------------- simülasyon

@dataclass
class Position:
    j: int
    side: int
    entry: float
    opened: int          # adım
    opened_ns: int
    signal_hour: int
    lot: float
    stop_pips: float
    cost_pips: float
    pv: float            # 1 lot pip değeri ($)
    atr: float
    target: float        # fiyat
    exit_level: float    # yarım ya da tam stop fiyatı
    margin: float
    sl: float = math.nan  # takip stopu
    trailing: bool = False
    best: float = math.nan
    against: int = 0     # ML'nin üst üste ters kaldığı saat başı sayısı


def hit_curve(stats: dict, pair: str) -> tuple[list[float], list[float], list[float]]:
    entry = stats["pairs"].get(pair) or stats["pooled"]
    rows = sorted((float(r), row) for r, row in entry["ratios"].items())
    return [r for r, _ in rows], [row["ml"]["target_rate"] for _, row in rows], \
        [(row["ml"]["target_rate"] + row["opposite"]["target_rate"]) / 2 for _, row in rows]


def ml_against(prob: float, side: int, direction: int, exit_conf: float) -> bool:
    """Saat başındaki ML okuması pozisyonun tersinde ve en az exit_conf güvende mi."""
    return (1 if prob > 0.5 else -1) * direction != side and abs(prob - 0.5) >= exit_conf - 0.5


def news_blocked(events: dict[str, np.ndarray], currencies: tuple[str, str], ns: int) -> bool:
    lo, hi = ns - 15 * 60 * 10**9, ns + 60 * 60 * 10**9  # haber 15 dk önce olduysa ya da 60 dk içinde olacaksa
    for currency in currencies:
        times = events.get(currency)
        if times is not None and times.size:
            k = np.searchsorted(times, lo)
            if k < times.size and times[k] <= hi:
                return True
    return False


def run(variant: Variant, market: dict, events: dict, stats: dict, steps: int | None = None,
        fixed_risk: bool = False) -> dict:
    """fixed_risk: risk ve teminat sınırı hep 500 $'a göre (bileşik yok); tüm dönemi ölçmek için ek görünüm."""
    pips = np.array([get_pip_size(p) for p in PAIRS])
    names = [(p[:3], p[3:6]) for p in PAIRS]
    curves = [hit_curve(stats, p) for p in PAIRS]
    O, H, L, A, P = (market[k] for k in ("Open", "High", "Low", "ATR", "Prob"))
    S15, S5, VOL = (market.get(k) for k in ("S15", "S5", "Vol"))
    ns, ny_min, ny_hour, ny_wd = market["ns"], market["ny_minutes"], market["ny_hour"], market["ny_weekday"]
    usd, spread = market["usd"], market["spread"]
    total = len(ns) if steps is None else min(steps, len(ns))
    hour_ns = 3600 * 10**9

    balance, trades, open_pos = BALANCE, [], []
    b_key, b_peak, b_peak_i, b_prev = None, -math.inf, 0, None  # sepet takibi
    last_loss: dict[int, int] = {}  # parite → son zararlı kapanış (ns)
    entered: set[tuple[int, int]] = set()
    equity_curve = []

    def pnl(pos: Position, price: float) -> float:
        return (pos.side * (price - pos.entry) / pips[pos.j] - pos.cost_pips) * pos.pv * pos.lot

    def close(pos: Position, i: int, price: float, reason: str) -> None:
        nonlocal balance
        gross = pos.side * (price - pos.entry) / pips[pos.j]
        net_usd = (gross - pos.cost_pips) * pos.pv * pos.lot
        balance += net_usd
        if net_usd < 0:
            last_loss[pos.j] = int(ns[i])
        trades.append({"pair": PAIRS[pos.j], "side": pos.side, "opened": pos.opened_ns, "closed": int(ns[i]),
                       "lot": pos.lot, "gross_pips": gross, "net_usd": net_usd, "reason": reason,
                       "trailing": pos.trailing, "risk_usd": (pos.stop_pips + pos.cost_pips) * pos.pv * pos.lot})
        open_pos.remove(pos)

    for i in range(total):
        t_ns = ns[i]
        new_hour = (t_ns % hour_ns) == 0
        # 1) açılışta: ML dönüşü, 24 saat, açılış boşluğu
        for pos in list(open_pos):
            o = O[i, pos.j]
            if not np.isfinite(o):
                continue
            if not pos.trailing:
                if variant.ml_exit and new_hour and np.isfinite(P[i, pos.j]) and P[i, pos.j] != 0.5:
                    pos.against = pos.against + 1 if ml_against(P[i, pos.j], pos.side, variant.direction, variant.exit_conf) else 0
                    if pos.against >= variant.exit_confirm:
                        close(pos, i, o, "ML YÖN DÖNDÜ")
                        continue
                if t_ns - pos.opened_ns >= HOLD_HOURS * hour_ns:
                    close(pos, i, o, "SÜRE")
                    continue
                if pos.side * (o - pos.exit_level) <= 0:
                    close(pos, i, o, "YARIM STOP" if variant.half_stop else "STOP")
                    continue
                if variant.use_target and not variant.trailing and pos.side * (o - pos.target) >= 0:
                    close(pos, i, o, "HEDEF")
                    continue
            elif pos.side * (o - pos.sl) <= 0:
                close(pos, i, o, "TAKİP STOPU")
                continue

        # 1b) sepet: toplam net kâr hedefi (açılış fiyatlarıyla)
        basket_target = (BALANCE if fixed_risk else balance) * variant.basket_pct
        if variant.basket_pct and open_pos and all(np.isfinite(O[i, pos.j]) for pos in open_pos):
            net = sum(pnl(pos, O[i, pos.j]) for pos in open_pos)
            if not variant.basket_trail:
                if net >= basket_target:
                    for pos in list(open_pos):
                        close(pos, i, O[i, pos.j], "TOPLAM HEDEF")
            else:
                key = frozenset(id(pos) for pos in open_pos)
                if key != b_key:
                    b_key, b_peak, b_peak_i, b_prev = key, net, i, None
                elif net > b_peak:
                    b_peak, b_peak_i = net, i
                gave_back = net <= b_peak * (1 - variant.basket_giveback)
                turning = i - b_peak_i >= TRAIL_STALL_STEPS and b_prev is not None and net < b_prev
                if b_peak >= basket_target and (gave_back or turning):
                    for pos in list(open_pos):
                        close(pos, i, O[i, pos.j], "SEPET TAKİP")
                    b_key = None
                else:
                    b_prev = net

        # 2) açılışta yeni girişler (ve gerekirse takas)
        weekday, minutes = ny_wd[i], ny_min[i]
        closed_market = weekday == 5 or (weekday == 6 and minutes < 17 * 60) or (weekday == 4 and minutes >= 16 * 60 + 45)
        rollover = 16 * 60 + 45 <= minutes < 18 * 60
        if not closed_market and not rollover and (balance > 0 or fixed_risk):
            pool = []
            if variant.swap and len(open_pos) >= MAX_POSITIONS:
                for pos in open_pos:
                    o = O[i, pos.j]
                    if pos.trailing or not np.isfinite(o) or t_ns - pos.opened_ns < STALE_HOURS * hour_ns:
                        continue
                    if pos.side * (o - pos.entry) / pips[pos.j] - pos.cost_pips >= 0:
                        progress = pos.side * (o - pos.entry) / abs(pos.target - pos.entry)
                        pool.append((progress, pos))
                pool.sort(key=lambda x: x[0])
            if len(open_pos) < MAX_POSITIONS or pool:
                sizing = BALANCE if fixed_risk else balance
                loss_usd = sizing * RISK_PCT / 100
                target_usd = loss_usd * TARGET_R
                signal_hour = int(t_ns // hour_ns)
                held = {pos.j for pos in open_pos}
                cands = []
                for j in range(len(PAIRS)):
                    if j in held or (j, signal_hour) in entered:
                        continue
                    o, atr, prob = O[i, j], A[i, j], P[i, j]
                    if not (np.isfinite(o) and np.isfinite(atr) and atr > 0 and np.isfinite(prob)) or prob == 0.5:
                        continue
                    side = (1 if prob > 0.5 else -1) * variant.direction
                    if variant.min_conf and abs(prob - 0.5) < variant.min_conf - 0.5:
                        continue
                    if variant.cooldown_hours and t_ns - last_loss.get(j, -10**19) < variant.cooldown_hours * hour_ns:
                        continue
                    if variant.need_15m and not side * S15[i, j] >= SUPPORT_SCORE:
                        continue
                    if variant.need_5m and not side * S5[i, j] >= SUPPORT_SCORE:
                        continue
                    if variant.vol and not (np.isfinite(VOL[i, j])
                                            and (VOL[i, j] >= VOL_THRESHOLD) == (variant.vol == 1)):
                        continue
                    pip = pips[j]
                    pv = pip * CONTRACT * usd[names[j][1]][i]
                    stop_pips = STOP_ATR * atr / pip
                    cost = spread[ny_hour[i], j] + ECN_COMMISSION_PIPS + variant.extra_cost
                    if cost / (stop_pips + cost) > variant.max_cost_share:
                        continue
                    lot = math.floor(loss_usd / ((stop_pips + cost) * pv) / MIN_LOT + 1e-9) * MIN_LOT
                    if lot < MIN_LOT:
                        continue
                    target_pips = target_usd / (lot * pv) + cost
                    ratios, ml_hits, any_hits = curves[j]
                    hit = float(np.interp(target_pips / stop_pips, ratios, ml_hits if variant.direction == 1 else any_hits))
                    if hit < MIN_HIT or cost / target_pips > MAX_COST_OF_TARGET:
                        continue
                    if news_blocked(events, names[j], t_ns):
                        continue
                    cands.append((hit, j, side, lot, stop_pips, cost, pv, target_pips))
                cands.sort(key=lambda c: -c[0])
                for hit, j, side, lot, stop_pips, cost, pv, target_pips in cands:
                    exposure: dict[str, int] = {}
                    for pos in open_pos:
                        for cur, d in ((names[pos.j][0], pos.side), (names[pos.j][1], -pos.side)):
                            exposure[cur] = exposure.get(cur, 0) + d
                    if any(exposure.get(cur, 0) * d >= MAX_CURRENCY for cur, d in ((names[j][0], side), (names[j][1], -side))):
                        continue
                    margin = lot * CONTRACT * usd[names[j][0]][i] / LEVERAGE
                    def fits() -> bool:
                        return (len(open_pos) < MAX_POSITIONS
                                and sum(p.margin for p in open_pos) + margin <= sizing * MAX_MARGIN_SHARE)
                    while not fits() and pool:
                        _, victim = pool.pop(0)
                        if victim in open_pos:
                            close(victim, i, O[i, victim.j], "DAHA İYİ FIRSAT")
                    if not fits():
                        continue
                    o, pip = O[i, j], pips[j]
                    half = EARLY_EXIT_SHARE if variant.half_stop else 1.0
                    open_pos.append(Position(
                        j=j, side=side, entry=o, opened=i, opened_ns=int(t_ns), signal_hour=signal_hour, lot=lot,
                        stop_pips=stop_pips, cost_pips=cost, pv=pv, atr=A[i, j], target=o + side * target_pips * pip,
                        exit_level=o - side * stop_pips * half * pip, margin=margin, best=o))
                    entered.add((j, signal_hour))

        # 3) mum içi: önce kötü taraf (aynı mumda ikisi de varsa stop sayılır)
        if variant.basket_pct and variant.basket_optimistic and open_pos:
            extremes = [(pos, H[i, pos.j] if pos.side == 1 else L[i, pos.j]) for pos in open_pos]
            if all(np.isfinite(x) for _, x in extremes) and sum(pnl(pos, x) for pos, x in extremes) >= basket_target:
                for pos, x in extremes:
                    close(pos, i, x, "TOPLAM HEDEF")
        for pos in list(open_pos):
            hi, lo = H[i, pos.j], L[i, pos.j]
            if not (np.isfinite(hi) and np.isfinite(lo)):
                continue
            worst, best = (lo, hi) if pos.side == 1 else (hi, lo)
            if not pos.trailing:
                if pos.side * (worst - pos.exit_level) <= 0:
                    close(pos, i, pos.exit_level, "YARIM STOP" if variant.half_stop else "STOP")
                    continue
                if variant.use_target and pos.side * (best - pos.target) >= 0:
                    if not variant.trailing:
                        close(pos, i, pos.target, "HEDEF")
                        continue
                    pos.trailing, pos.best = True, best
                    if variant.floor_atr is None:
                        floor = pos.entry + pos.side * pos.cost_pips * pips[pos.j]
                    elif math.isinf(variant.floor_atr):
                        floor = pos.exit_level
                    else:
                        floor = pos.entry - pos.side * variant.floor_atr * pos.atr
                    trail = best - pos.side * TRAIL_ATR * pos.atr
                    pos.sl = max(floor, trail) if pos.side == 1 else min(floor, trail)
                    if pos.side * (worst - pos.sl) <= 0:  # aynı mumda geri döndü: başabaşta çık
                        close(pos, i, pos.sl, "TAKİP STOPU")
                    continue
            else:
                if pos.side * (worst - pos.sl) <= 0:
                    close(pos, i, pos.sl, "TAKİP STOPU")
                    continue
                if pos.side * (best - pos.best) > 0:
                    pos.best = best
                    trail = best - pos.side * TRAIL_ATR * pos.atr
                    pos.sl = max(pos.sl, trail) if pos.side == 1 else min(pos.sl, trail)
        if i % 288 == 0:
            equity_curve.append((int(t_ns), balance))
    for pos in list(open_pos):  # dönem sonunda açık kalanlar son açılışta kapanır
        last = total - 1
        close(pos, last, O[last, pos.j] if np.isfinite(O[last, pos.j]) else pos.entry, "DÖNEM SONU")
    equity_curve.append((int(ns[total - 1]), balance))
    return {"variant": variant.name, "trades": trades, "equity": equity_curve}


def ruin_dates(equity: pd.Series) -> dict:
    out = {}
    for label, level in (("half", BALANCE / 2), ("below_100", 100.0)):
        below = equity[equity < level]
        out[label] = str(below.index[0].date()) if not below.empty else None
    return out


def summarize(result: dict) -> dict:
    trades = pd.DataFrame(result["trades"])
    equity = pd.Series([b for _, b in result["equity"]],
                       index=pd.to_datetime([t for t, _ in result["equity"]], utc=True))
    if trades.empty:
        return {"variant": result["variant"], "trades": 0}
    trades["year"] = pd.to_datetime(trades["closed"], utc=True).dt.year
    peak = equity.cummax()
    drawdown = float(((equity - peak) / peak).min())
    losing = (trades["net_usd"] < 0).astype(int)
    streak = int(losing.groupby((losing == 0).cumsum()).sum().max())
    r = trades["net_usd"] / trades["risk_usd"]
    trades["r"] = r
    by_reason = trades.groupby("reason").agg(count=("net_usd", "size"), mean=("net_usd", "mean"), sum=("net_usd", "sum"),
                                             median_r=("r", "median")).round(3)
    by_year = trades.groupby("year").agg(n=("net_usd", "size"), net=("net_usd", "sum"), win=("net_usd", lambda s: (s > 0).mean()))
    return {
        "variant": result["variant"], "trades": int(len(trades)), "final_balance": float(equity.iloc[-1]),
        "return_pct": float(equity.iloc[-1] / BALANCE * 100 - 100), "max_drawdown_pct": drawdown * 100,
        "win_rate": float((trades["net_usd"] > 0).mean()), "mean_r": float(r.mean()),
        "trailing_rate": float(trades["trailing"].mean()), "longest_losing_streak": streak,
        "by_reason": {k: v for k, v in by_reason.to_dict("index").items()},
        "by_year": {int(k): {"n": int(v["n"]), "net": float(v["net"]), "win": float(v["win"])} for k, v in by_year.iterrows()},
        "trades_per_week": float(len(trades) / max((equity.index[-1] - equity.index[0]).days / 7, 1)),
        "ruin": ruin_dates(equity), "net_usd": float(trades["net_usd"].sum()),
        "mean_r_choose": float(r[trades["closed"] < CHOOSE_END.value].mean()),
        "mean_r_confirm": float(r[trades["closed"] >= CHOOSE_END.value].mean()),
    }


def _run_one(args) -> dict:
    variant, steps, fixed = args
    variants = VARIANTS + BASKET_VARIANTS + [Variant(**{**v.__dict__, "name": v.name + " (iyimser)", "basket_optimistic": True})
                                             for v in BASKET_VARIANTS] + TRAIL_VARIANTS + SUPPORT_VARIANTS + FLOOR_VARIANTS + CONF_VARIANTS
    started = time.time()
    market, events, stats = load_market(parallel=False), high_impact_events(), load_plan_stats()
    result = run(variant, market, events, stats, steps, fixed_risk=fixed)
    summary = summarize(result)
    summary["seconds"] = round(time.time() - started)
    tag = f"{variants.index(variant)}{'_fixed' if fixed else ''}"
    pd.DataFrame(result["trades"]).to_csv(CACHE / f"trades_{tag}.csv.gz", index=False)
    pd.DataFrame(result["equity"], columns=["ns", "balance"]).to_csv(CACHE / f"equity_{tag}.csv", index=False)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--fixed-risk", action="store_true", help="Ek görünüm: her işlemde 12,5 $ risk, bileşik yok")
    parser.add_argument("--basket", action="store_true", help="protocol_basket.json varyantları (temkinli + iyimser)")
    parser.add_argument("--basket-trail", action="store_true", help="protocol_basket_trail.json ızgarası ve kontrolleri")
    parser.add_argument("--support", action="store_true", help="protocol_support.json: 15M/5M/oynaklık desteği")
    parser.add_argument("--floor", action="store_true", help="protocol_trail_floor.json: takip tabanı")
    parser.add_argument("--confidence", action="store_true", help="protocol_confidence_cooldown.json")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    started = time.time()
    load_market()  # önbelleği bir kez doldur
    print(f"Veri hazır: {time.time() - started:.0f} sn", flush=True)
    if args.pilot:
        steps = 288 * 90
        summary = _run_one((TRAIL_VARIANTS[2] if args.basket_trail else VARIANTS[0], steps, args.fixed_risk))
        print(json.dumps(summary, ensure_ascii=False, indent=1, default=str))
        return
    chosen = VARIANTS
    if args.basket:
        chosen = BASKET_VARIANTS + [Variant(**{**v.__dict__, "name": v.name + " (iyimser)", "basket_optimistic": True})
                                    for v in BASKET_VARIANTS]
    if args.basket_trail:
        chosen = TRAIL_VARIANTS
    if args.support:
        chosen = SUPPORT_VARIANTS
    if args.floor:
        chosen = FLOOR_VARIANTS
    if args.confidence:
        chosen = CONF_VARIANTS
    with Pool(min(len(chosen), WORKERS if args.basket_trail or args.support or args.floor or args.confidence else 4)) as pool:
        summaries = pool.map(_run_one, [(v, None, args.fixed_risk) for v in chosen])
    name = ("results_confidence" if args.confidence else "results_floor" if args.floor else "results_support" if args.support else "results_basket_trail" if args.basket_trail else "results_basket" if args.basket else "results")         + ("_fixed_risk" if args.fixed_risk else "")
    (OUT / f"{name}.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    for s in summaries:
        print(f"{s['variant']:<36} işlem {s['trades']:>5} · son bakiye {s.get('final_balance', 0):>8.2f} $ "
              f"({s.get('return_pct', 0):+.1f}%) · en büyük düşüş {s.get('max_drawdown_pct', 0):.1f}% · "
              f"kazanma {s.get('win_rate', 0) * 100:.0f}% · ort. {s.get('mean_r', 0):+.3f} R "
              f"(2023–24 {s.get('mean_r_choose', 0):+.3f} · 2025–26 {s.get('mean_r_confirm', 0):+.3f}) · {s.get('seconds')} sn")


if __name__ == "__main__":
    main()
