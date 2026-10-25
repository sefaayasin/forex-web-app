"""Can the frozen direction signals take a small net profit before the original stop?

No new entries or trained model: reuse all 19 decisions, with independent
non-overlap accounting per target. This is a follow-up scenario on already
inspected data, not validation or a target-selection exercise.
"""
from pathlib import Path
import json

import pandas as pd

from forex_config import get_pip_size
from research_pivot_advisor import ROOT, read_bars, simulate, digest

OUT = ROOT / "research/pivot_advisor_small_profit"
DECISIONS = ROOT / "research/pivot_advisor/all_decisions.csv"
SCENARIOS = [(f"net_{p}_pip", "pip", p) for p in (1, 3, 5)] + [
    ("net_10_usd_risk_100", "cash", 10), ("net_50_usd_risk_100", "cash", 50)]


def target_distance(net_profit, cost_pips, cash_per_pip):
    return cost_pips + net_profit / cash_per_pip


def evaluate_scenario(decisions, bars_by_pair, mode, amount):
    records, counts = [], {"decisions": len(decisions), "overlaps": 0, "incomplete": 0, "invalid_open": 0}
    busy = {}
    for _, row in decisions.sort_values("time").iterrows():
        now = pd.Timestamp(row.time)
        if row.pair in busy and now < busy[row.pair]:
            counts["overlaps"] += 1
            continue
        bars = bars_by_pair[row.pair]
        pos = bars.index.searchsorted(now)
        if pos >= len(bars) or bars.index[pos] != now:
            counts["incomplete"] += 1
            continue
        pip = get_pip_size(row.pair)
        sign = 1 if row.side == "LONG" else -1
        planned_loss_pips = abs(row.signal_price - row.stop) / pip + row.cost_pips
        # Cash mode is only normalized risk arithmetic, not a broker lot simulation.
        cash_per_pip = 100 / planned_loss_pips if mode == "cash" else 1.
        actual_entry = float(bars.Open.iloc[pos])
        target = actual_entry + sign * target_distance(amount, row.cost_pips, cash_per_pip) * pip
        outcome = simulate(bars, now, row.side, row.signal_price, row.stop, target, pip)
        if outcome is None:
            counts["incomplete"] += 1
            continue
        if outcome["kind"] == "invalid_open":
            counts["invalid_open"] += 1
            continue
        net_pips = outcome["gross_pips"] - row.cost_pips
        records.append({"pair": row.pair, "time": now, "side": row.side, "requested_net": amount,
                        "unit": "USD scenario" if mode == "cash" else "pip", "target": target, "stop": row.stop,
                        "cost_pips": row.cost_pips, **outcome, "net_pips": net_pips,
                        "net_result": net_pips * cash_per_pip,
                        "stress_net_result": (net_pips - row.cost_pips - .5) * cash_per_pip})
        busy[row.pair] = outcome["exit_time"]
    return pd.DataFrame(records), counts


def summarize(trades):
    if trades.empty:
        return {"n": 0}
    return {"n": len(trades), "targets": int(trades.kind.eq("target").sum()),
            "stops": int(trades.kind.isin(["stop", "stop_gap"]).sum()),
            "timeouts": int(trades.kind.eq("time").sum()),
            "net_positive": int(trades.net_result.gt(1e-8).sum()),
            "net_negative": int(trades.net_result.lt(-1e-8).sum()),
            "net_positive_rate": float(trades.net_result.gt(1e-8).mean()),
            "mean_result": float(trades.net_result.mean()), "sum_result": float(trades.net_result.sum()),
            "mean_stress_result": float(trades.stress_net_result.mean()),
            "worst_result": float(trades.net_result.min())}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    specification = {"source": str(DECISIONS.relative_to(ROOT)), "decisions_sha256": digest(DECISIONS),
                     "code_sha256": digest(Path(__file__)), "scenarios": SCENARIOS,
                     "stop": "original pivot stop", "horizon_hours": 4,
                     "selection": "none; follow-up on inspected data, not validation",
                     "cash_policy": "illustrative $100 planned stop, continuous pip value, no lot/margin simulation",
                     "fills": "next 5M open; target calculated from actual fill; stop first if ambiguous"}
    protocol = OUT / "protocol.json"
    if protocol.exists() and json.loads(protocol.read_text(encoding="utf-8")) != json.loads(json.dumps(specification)):
        raise ValueError("Existing scenario protocol differs")
    protocol.write_text(json.dumps(specification, ensure_ascii=False, indent=2), encoding="utf-8")
    decisions = pd.read_csv(DECISIONS)
    start = pd.to_datetime(decisions.time, utc=True).min() - pd.Timedelta(days=1)
    bars = {pair: read_bars(pair, "historical_5m", start) for pair in decisions.pair.unique()}
    results = {}
    lines = ["# Küçük net kârı önce alabilir miydi?", "",
             "Aynı geçmiş fiyatlar ve aynı 19 GİR kararı. Yeni yön modeli veya yeni sinyal üretilmedi.",
             "Her hedef için açık işlem çakışması yeniden hesaplandı; küçük hedef erken kapanırsa sonraki sinyal alınabilir.",
             "Stop eski plandaki stop; üst süre 4 saat. Maliyet hedef mesafesine eklendi.", "",
             "| Net hedef | İşlem | Hedef | Stop | Süre sonu | Net kâr/zarar | Ortalama net sonuç |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name, mode, amount in SCENARIOS:
        trades, audit = evaluate_scenario(decisions, bars, mode, amount)
        trades.to_csv(OUT / f"{name}_trades.csv", index=False)
        result = {"mode": mode, "amount": amount, "all": summarize(trades), "audit": audit,
                  "by_side": {side: summarize(trades[trades.side == side]) for side in ("LONG", "SHORT")}}
        results[name] = result
        s = result["all"]
        label = f"{amount} pip" if mode == "pip" else f"${amount} / $100 planlanan stop"
        lines.append(f"| {label} | {s['n']} | {s['targets']} | {s['stops']} | {s['timeouts']} | "
                     f"{s['net_positive']} / {s['net_negative']} | {s['mean_result']:+.2f} {'pip' if mode == 'pip' else '$'} |")
        print(name, json.dumps(result, ensure_ascii=False), flush=True)
    lines += ["", "Dolar satırları örnek $100 stop bütçesine göre matematiksel ölçeklemedir; gerçek hesabın getirisi değildir.",
              "Lot adımı, teminat, kur çevrimi ve değişen spread dahil edilmedi; broker kotasyonu yerine BID OHLC kullanıldı.",
              "Daha yüksek spread/komisyon ve ek 0,5 pip kayma sonuçları results.json içinde ayrıca gösterilir.",
              "Örnek çok küçük ve daha önce incelenmiş veridir. Yüksek hedefe ulaşma oranı, zarar etmeme garantisi vermez.",
              "Sinyalin sonradan doğru yönde gitmesi yerine, küçük net hedefe stop öncesi ulaşması ölçüldü."]
    (OUT / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
