"""Pure decision rules shared by live analysis and backtests.

This module intentionally has no Streamlit, network, or market-data dependency so
the trading decision can be regression-tested without starting the web app.
"""

from __future__ import annotations

import math
import random


def classify_opportunity_readiness(
    radar_score: float,
    side: str,
    catalyst_matches: bool,
    structure_matches: bool,
    htf_conflict: bool,
    capacity_ratio: float,
    in_session: bool,
    candidate_threshold: float = 42.0,
    ready_threshold: float = 65.0,
) -> tuple[str, list[str]]:
    """Classify a radar snapshot without pretending its score is a probability.

    READY is deliberately reserved for a directional trigger with compatible
    structure, higher timeframes, session and realistic recent range.  WATCH is
    informative but must never be interpreted as permission to place an order.
    """
    normalized_side = str(side).upper()
    score = float(radar_score) if not _missing(radar_score) else 0.0
    if normalized_side not in {"LONG", "SHORT"} or score < float(candidate_threshold):
        return "NEUTRAL", ["Radar puanı aday eşiğinin altında"]

    blockers: list[str] = []
    if score < float(ready_threshold):
        blockers.append(f"Radar puanı {score:.0f}; hazır eşiği {float(ready_threshold):.0f}")
    if not catalyst_matches:
        blockers.append("15M tepki veya Bollinger kırılım tetiği yok")
    if not structure_matches:
        blockers.append("15M MA ve swing yapısı yönü doğrulamıyor")
    if htf_conflict:
        blockers.append("4H ve 1H yönleri çelişiyor")
    if _missing(capacity_ratio):
        blockers.append("Hedef kapasitesi ölçülemedi")
    elif float(capacity_ratio) > 0.80:
        blockers.append("Hedef tipik 4 saatlik hareketin %80'inden büyük")
    if not in_session:
        blockers.append("Seçili likit işlem seansı dışında")

    return ("WATCH", blockers) if blockers else ("READY", [])


def _missing(value: float) -> bool:
    try:
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return True


def decide_mtf_signal(
    entry_score: float,
    h4_score: float,
    h1_score: float,
    m15_score: float,
    tf_name: str,
    threshold: float,
) -> tuple[str, str]:
    """Return LONG/SHORT/NONE using the canonical closed-bar MTF rule."""
    if _missing(entry_score) or _missing(h4_score) or _missing(h1_score):
        return "NONE", "Ana zaman dilimi skorları yetersiz"

    htf_long = float(h4_score) >= 25 and float(h1_score) >= 25
    htf_short = float(h4_score) <= -25 and float(h1_score) <= -25

    if tf_name == "5 Dakika":
        m15_long_ok = not _missing(m15_score) and float(m15_score) >= 25
        m15_short_ok = not _missing(m15_score) and float(m15_score) <= -25
    else:
        m15_long_ok = True
        m15_short_ok = True

    if htf_long and m15_long_ok and float(entry_score) >= float(threshold):
        return "LONG", "4H+1H long uyumlu; giriş skoru eşiği geçti"
    if htf_short and m15_short_ok and float(entry_score) <= -float(threshold):
        return "SHORT", "4H+1H short uyumlu; giriş skoru eşiği geçti"

    return "NONE", "MTF filtre veya giriş skoru uygun değil"


def position_level_event(side: str, current_price: float, stop: float, target: float) -> str:
    """Return STOP/TARGET/NONE without mixing LONG and SHORT inequalities."""
    normalized_side = str(side).upper()
    current = float(current_price)
    stop_value = float(stop)
    target_value = float(target)
    if normalized_side == "LONG":
        if stop_value > 0 and current <= stop_value:
            return "STOP"
        if target_value > 0 and current >= target_value:
            return "TARGET"
    elif normalized_side == "SHORT":
        if stop_value > 0 and current >= stop_value:
            return "STOP"
        if target_value > 0 and current <= target_value:
            return "TARGET"
    return "NONE"


def _finite_values(values) -> list[float]:
    result: list[float] = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            result.append(number)
    return result


def _percentile(values: list[float], probability: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    position = min(max(float(probability), 0.0), 1.0) * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def stationary_bootstrap_mean_test(
    values,
    simulations: int = 2000,
    mean_block_length: float = 5.0,
    seed: int = 42,
) -> dict:
    """One-sided mean-edge test that preserves short-range serial dependence.

    The confidence interval is bootstrapped from the observed series.  The
    p-value is computed from a demeaned null series, so it answers whether the
    observed mean is unusually positive under a zero-edge hypothesis.
    """
    clean = _finite_values(values)
    n = len(clean)
    simulations = max(int(simulations), 100)
    if n < 5:
        return {
            "status": "insufficient",
            "sample_count": n,
            "observed_mean": math.nan if not clean else sum(clean) / n,
            "ci_low": math.nan,
            "ci_high": math.nan,
            "p_value": math.nan,
        }

    observed_mean = sum(clean) / n
    centered = [value - observed_mean for value in clean]
    restart_probability = min(1.0, max(1.0 / float(mean_block_length), 1.0 / n))
    rng = random.Random(int(seed))
    raw_means: list[float] = []
    null_means: list[float] = []

    for _ in range(simulations):
        index = rng.randrange(n)
        raw_total = 0.0
        null_total = 0.0
        for position in range(n):
            if position > 0:
                if rng.random() < restart_probability:
                    index = rng.randrange(n)
                else:
                    index = (index + 1) % n
            raw_total += clean[index]
            null_total += centered[index]
        raw_means.append(raw_total / n)
        null_means.append(null_total / n)

    p_value = (1 + sum(value >= observed_mean for value in null_means)) / (simulations + 1)
    return {
        "status": "ready",
        "sample_count": n,
        "observed_mean": observed_mean,
        "ci_low": _percentile(raw_means, 0.025),
        "ci_high": _percentile(raw_means, 0.975),
        "p_value": p_value,
        "simulations": simulations,
        "mean_block_length": float(mean_block_length),
    }


def circular_shift_timing_test(
    entry_prices,
    exit_prices,
    entry_indices,
    side_signs,
    horizon_bars: int,
    pip_size: float,
    cost_pips: float = 0.0,
    simulations: int = 2000,
    seed: int = 42,
) -> dict:
    """Compare actual signal timing with circularly shifted signal schedules.

    A global shift keeps signal spacing and LONG/SHORT ordering intact while
    breaking their alignment with subsequent market moves.  The statistic is
    average fixed-horizon net pips, separate from the strategy's exit model.
    """
    entries = _finite_values(entry_prices)
    exits = _finite_values(exit_prices)
    if len(entries) != len(exits):
        return {"status": "invalid", "sample_count": 0, "p_value": math.nan}
    horizon = max(int(horizon_bars), 1)
    available = len(entries) - horizon
    if available < 10 or not math.isfinite(float(pip_size)) or float(pip_size) <= 0:
        return {"status": "insufficient", "sample_count": 0, "p_value": math.nan}

    schedule: list[tuple[int, float]] = []
    for raw_index, raw_side in zip(entry_indices, side_signs):
        try:
            index = int(raw_index)
            side = float(raw_side)
        except (TypeError, ValueError):
            continue
        if 0 <= index < available and side != 0 and math.isfinite(side):
            schedule.append((index, 1.0 if side > 0 else -1.0))
    if len(schedule) < 5:
        return {"status": "insufficient", "sample_count": len(schedule), "p_value": math.nan}

    pip = float(pip_size)
    cost = max(float(cost_pips), 0.0)

    def schedule_mean(shift: int) -> float:
        total = 0.0
        for index, side in schedule:
            shifted = (index + shift) % available
            total += side * (exits[shifted + horizon] - entries[shifted]) / pip - cost
        return total / len(schedule)

    observed_mean = schedule_mean(0)
    simulations = max(int(simulations), 100)
    rng = random.Random(int(seed))
    null_means = [schedule_mean(rng.randrange(1, available)) for _ in range(simulations)]
    p_value = (1 + sum(value >= observed_mean for value in null_means)) / (simulations + 1)
    percentile = (1 + sum(value <= observed_mean for value in null_means)) / (simulations + 1)
    return {
        "status": "ready",
        "sample_count": len(schedule),
        "observed_mean_pips": observed_mean,
        "null_median_pips": _percentile(null_means, 0.50),
        "null_p95_pips": _percentile(null_means, 0.95),
        "p_value": p_value,
        "percentile": percentile,
        "horizon_bars": horizon,
        "simulations": simulations,
    }


def bonferroni_adjust(p_value: float, trials: int) -> float:
    if _missing(p_value):
        return math.nan
    return min(max(float(p_value), 0.0) * max(int(trials), 1), 1.0)


def classify_edge_evidence(
    trade_count: int,
    average_r: float,
    r_ci_low: float,
    bootstrap_p_adjusted: float,
    oos_trade_count: int,
    oos_average_r: float,
    min_trades: int = 60,
    min_oos_trades: int = 12,
    alpha: float = 0.05,
) -> tuple[str, list[str]]:
    """Classify expectancy evidence without making timing a duplicate hard gate.

    The primary claim is that the complete strategy has positive expectancy.
    Entry timing is reported separately because it tests a different hypothesis
    and is computed from the same market sample.
    """
    blockers: list[str] = []
    if int(trade_count) < int(min_trades):
        blockers.append(f"İşlem sayısı {int(trade_count)}; minimum {int(min_trades)}")
    if _missing(average_r) or float(average_r) <= 0:
        blockers.append("Ortalama R pozitif değil")
    if _missing(r_ci_low) or float(r_ci_low) <= 0:
        blockers.append("Ortalama R için %95 alt güven sınırı sıfırın üstünde değil")
    if _missing(bootstrap_p_adjusted) or float(bootstrap_p_adjusted) > float(alpha):
        blockers.append("Bootstrap edge testi çoklu-deneme düzeltmesinden geçmedi")
    if int(oos_trade_count) < int(min_oos_trades):
        blockers.append(f"Son %30 OOS işlem sayısı {int(oos_trade_count)}; minimum {int(min_oos_trades)}")
    if _missing(oos_average_r) or float(oos_average_r) <= 0:
        blockers.append("Son %30 OOS ortalama R pozitif değil")

    if not blockers:
        return "DOĞRULANDI", []

    candidate_min_trades = max(30, int(math.ceil(float(min_trades) * 2 / 3)))
    candidate_min_oos = max(8, int(math.ceil(float(min_oos_trades) * 2 / 3)))
    candidate = (
        int(trade_count) >= candidate_min_trades
        and not _missing(average_r)
        and float(average_r) > 0
        and not _missing(bootstrap_p_adjusted)
        and float(bootstrap_p_adjusted) <= 0.10
        and int(oos_trade_count) >= candidate_min_oos
        and not _missing(oos_average_r)
        and float(oos_average_r) > 0
    )
    if candidate:
        return "ADAY / DEMO", blockers
    if int(trade_count) < candidate_min_trades:
        return "YETERSİZ ÖRNEK", blockers
    return "DOĞRULANMADI", blockers
