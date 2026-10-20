"""Small-profit exit plan around the frozen v2 direction selector. Never sends orders."""
from __future__ import annotations

import math
import pandas as pd

from forex_lot_calculator import floor_lot
from forex_pivot_advisor import advise, finite

NET_TARGET_PIPS = 3.0
MAX_QUOTE_AGE_SECONDS = 120
MAX_ANALYSIS_AGE_SECONDS = 90
MAX_HOLD_HOURS = 4


def utc(value):
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def live_plan(snapshot, quote, *, now, news_known, news_minutes=None, news_title="",
              cost_pips=None, account_usd=1000., max_loss_usd=5., pip_value_per_lot=None,
              anchor_price=None):
    """Keep the original pivot selection gates, replace only its exit by net 3 pip.

    Freshness, execution-cost and price-drift checks are additional live checks;
    the 12 historical examples are not validation of these added checks.
    Prices are BID/indicative references. MT5 broker SL/TP are converted for
    SHORT's ASK execution; commission is included in cost_pips by the caller.
    """
    quote = quote or {}
    price = quote.get("price")
    work = dict(snapshot)
    if finite(price):
        work["price"] = float(price)
        if all(finite(work.get(k)) for k in ("entry", "atr")) and work["atr"] > 0:
            work["distance_atr"] = abs(float(price) - work["entry"]) / work["atr"]
    result = advise(work, now=now, news_known=news_known, news_minutes=news_minutes,
                    news_title=news_title, cost_pips=cost_pips)
    blockers = list(result["blockers"])
    provider = snapshot.get("provider", "")
    if provider == "Broker CSV":
        blockers.append("CSV geçmiş veridir; canlı giriş için güncel fiyat kaynağı seç.")
    for stamp, maximum, label in ((quote.get("time"), MAX_QUOTE_AGE_SECONDS, "Anlık fiyat"),
                                   (snapshot.get("as_of"), MAX_ANALYSIS_AGE_SECONDS, "Analiz")):
        try:
            age = (now - utc(stamp)).total_seconds() if stamp is not None else math.inf
        except (TypeError, ValueError):
            age = math.inf
        if not 0 <= age <= maximum:
            blockers.append(f"{label} eski veya alınamadı; yenilemeden giriş yapma.")
    if not finite(price) or float(price) <= 0:
        blockers.append("Geçerli güncel fiyat alınamadı.")
    if snapshot.get("side") in {"LONG", "SHORT"} and all(finite(v) for v in (price, snapshot.get("entry"))):
        sign = 1 if snapshot["side"] == "LONG" else -1
        if sign * (float(price) - float(snapshot["entry"])) <= 0:
            blockers.append("Fiyat pivot giriş seviyesinin diğer tarafına geçti; eski planla giriş yapma.")
    anchor = anchor_price if finite(anchor_price) else snapshot.get("price")
    pip, atr = snapshot.get("pip"), snapshot.get("atr")
    if all(finite(v) for v in (price, anchor, pip, atr)) and pip > 0 and atr > 0:
        drift_limit = min(.25 * atr, .5 * NET_TARGET_PIPS * pip)
        if abs(float(price) - float(anchor)) > drift_limit + 1e-12:
            blockers.append("Fiyat ilk giriş bölgesinden uzaklaştı; hareketi kovalama, yeni tepkiyi bekle.")
    else:
        blockers.append("Giriş bölgesi ve fiyat değişimi hesaplanamadı.")
    valid_budget = all(finite(v) for v in (account_usd, max_loss_usd, pip_value_per_lot))
    if not valid_budget or min(account_usd, max_loss_usd, pip_value_per_lot) <= 0 or max_loss_usd > account_usd:
        blockers.append("Pozitif hesap, zarar bütçesi ve USD pip değeri gerekli; sıfır riskle pozisyon açılamaz.")
    side = snapshot.get("side")
    geometry = side in {"LONG", "SHORT"} and all(finite(v) for v in (price, pip, result.get("stop"), cost_pips))
    lot, target, loss, profit, gross_target_pips, rr = 0., None, None, None, None, None
    executable_entry = broker_stop = broker_target = None
    if geometry and pip > 0 and cost_pips >= 0:
        sign = 1 if side == "LONG" else -1
        stop_pips = sign * (float(price) - result["stop"]) / pip
        gross_target_pips = NET_TARGET_PIPS + cost_pips
        target = float(price) + sign * gross_target_pips * pip
        if stop_pips > 0 and valid_budget and min(account_usd, max_loss_usd, pip_value_per_lot) > 0:
            lot = floor_lot(min(max_loss_usd, account_usd) / ((stop_pips + cost_pips) * pip_value_per_lot))
            loss = (stop_pips + cost_pips) * lot * pip_value_per_lot
            profit = NET_TARGET_PIPS * lot * pip_value_per_lot
            rr = NET_TARGET_PIPS / (stop_pips + cost_pips)
            if lot < .01:
                blockers.append("Zarar bütçesi en küçük 0,01 lota yetmiyor; bu planla giriş yapma.")
        # BID reference includes the spread once in total cost. For a SHORT,
        # the broker triggers SL/TP against ASK, so shift both levels by spread.
        bid, ask = quote.get("bid"), quote.get("ask")
        if all(finite(v) for v in (bid, ask)) and ask >= bid > 0:
            spread = float(ask) - float(bid)
            executable_entry = float(ask) if side == "LONG" else float(bid)
            broker_stop = result["stop"] + (spread if side == "SHORT" else 0)
            broker_target = target + (spread if side == "SHORT" else 0)
    blockers = list(dict.fromkeys(blockers))
    ready = not blockers and target is not None and lot >= .01
    action = f"{side} GİR" if ready else "BEKLE"
    valid_until = None
    if quote.get("time") is not None and snapshot.get("as_of") is not None:
        try:
            valid_until = min(utc(quote["time"]) + pd.Timedelta(seconds=MAX_QUOTE_AGE_SECONDS),
                              utc(snapshot["as_of"]) + pd.Timedelta(seconds=MAX_ANALYSIS_AGE_SECONDS))
        except (TypeError, ValueError):
            pass
    return {**result, "action": action, "technical_ready": ready, "blockers": blockers,
            "pivot_target": result.get("target"), "target": target, "entry": price,
            "net_target_pips": NET_TARGET_PIPS, "gross_target_pips": gross_target_pips,
            "small_target_rr": rr, "lot": lot, "planned_loss_usd": loss, "expected_profit_usd": profit,
            "quote_time": quote.get("time"), "valid_until": valid_until,
            "broker_entry": executable_entry, "broker_stop": broker_stop, "broker_target": broker_target,
            "comment": (f"{action}: ana trend ve kapanmış tepki uyumlu; giriş bölgesi, haber ve maliyet kontrolü geçti. "
                        f"Tahmini maliyet sonrası 3 pipte çıkış planı: yaklaşık +{profit:.2f} $, "
                        f"stopta planlanan kayıp −{loss:.2f} $." if ready else
                        "Şu an giriş önerisi yok. " + (blockers[0] if blockers else "Plan hesaplanamadı.")),
            "performance_note": "Küçük örnek üzerinde geçmiş deneme; gelecekteki başarı ihtimali veya zarar etmeme garantisi değildir."}
