"""ML proof-of-concept: does a gradient-boosted classifier add edge on top of
technical + macro/news features for EURUSD 1H, beyond what the existing
rule-based decision core already captures?

This is deliberately a *support* signal, not a replacement: it outputs a
probability that is meant to gate/filter the existing READY/WATCH/NEUTRAL
logic in forex_decision_core.py, the same way the statistical edge validation
in forex_edge.py gates trade-taking.

Design choices, and why:
  - Time-ordered train/test split only (no shuffling, no k-fold on raw rows):
    shuffling would leak future bars into training via overlapping indicator
    windows and the triple-barrier label horizon.
  - Triple-barrier labeling (ATR-scaled long/short barriers over a fixed bar
    horizon) instead of a plain "close > close[t+N]" label, so the target
    matches something a rule-based strategy could actually trade rather than
    an arbitrary point-in-time return.
  - Edge is judged the same way forex_edge.py judges strategies: mean R of
    the label outcome inside the model's high-confidence subset, tested with
    the existing stationary-bootstrap significance test, not raw accuracy.
    Accuracy on an imbalanced/near-50 label is not evidence of a tradeable
    edge by itself.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from forex_decision_core import bonferroni_adjust, stationary_bootstrap_mean_test
from forex_indicators import add_indicators

SYMBOL = "EURUSD"
BAR_HOURS = 1
HORIZON_BARS = 24          # ~1 trading day ahead
BARRIER_ATR_MULT = 1.5     # favorable/adverse move size, in ATR units
TEST_FRACTION = 0.2        # final chronological slice held out
CONFIDENCE_THRESHOLD = 0.60


def load_ohlc(symbol: str) -> pd.DataFrame:
    df = pd.read_csv(f"data/historical_1h/{symbol}.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp").sort_index()
    df = df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"})
    return df[["Open", "High", "Low", "Close", "Volume"]]


def load_macro_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Daily/weekly macro & news series, forward-filled onto the bar index.

    Forward-fill only (never back-fill): a value must not be visible before
    its real publication date, or the model would be trained on information
    it could not have had at decision time.
    """
    dates = index.tz_convert("UTC").normalize()
    out = pd.DataFrame(index=index)

    unique_dates = dates.unique().sort_values()
    fed = pd.read_csv("data/news/fred/DFEDTARU.csv", parse_dates=["date"]).set_index("date")["value"]
    ecb = pd.read_csv("data/news/fred/ECBDFR.csv", parse_dates=["date"]).set_index("date")["value"]
    fed.index = fed.index.tz_localize("UTC")
    ecb.index = ecb.index.tz_localize("UTC")
    fed_on_dates = fed.reindex(unique_dates).sort_index().ffill()
    ecb_on_dates = ecb.reindex(unique_dates).sort_index().ffill()
    rate_diff = (ecb_on_dates - fed_on_dates).reindex(dates).to_numpy()
    out["rate_diff_ecb_fed"] = rate_diff

    fomc = pd.read_csv("data/news/fomc_statements.csv", parse_dates=["date"]).set_index("date")
    fomc.index = fomc.index.tz_localize("UTC")
    fomc = fomc.sort_index()
    sentiment_on_dates = fomc["sentiment_score"].reindex(unique_dates).sort_index().ffill()
    out["fomc_sentiment"] = sentiment_on_dates.reindex(dates).to_numpy()
    last_fomc_date = pd.Series(fomc.index, index=fomc.index).reindex(unique_dates).sort_index().ffill()
    days_since_fomc = (pd.Series(unique_dates, index=unique_dates) - last_fomc_date).dt.days
    out["days_since_fomc"] = days_since_fomc.reindex(dates).to_numpy()

    cot = pd.read_csv("data/news/cot/cot_currency_positioning.csv", parse_dates=["date"])
    eur_cot = cot[cot["currency"] == "EUR"].set_index("date").sort_index()
    eur_cot.index = eur_cot.index.tz_localize("UTC")
    net_z = (eur_cot["noncommercial_net"] - eur_cot["noncommercial_net"].rolling(52, min_periods=8).mean()) / (
        eur_cot["noncommercial_net"].rolling(52, min_periods=8).std().replace(0, np.nan)
    )
    net_on_dates = net_z.reindex(unique_dates).sort_index().ffill()
    out["eur_cot_net_z"] = net_on_dates.reindex(dates).to_numpy()

    events = pd.read_csv("data/news/events_with_impact.csv", parse_dates=["date"])
    events = events[events["currency"].isin(["EUR", "USD"])].drop_duplicates(subset=["date", "currency", "event"])
    events["date"] = events["date"].dt.tz_localize("UTC")
    for ccy in ("EUR", "USD"):
        ccy_events = events[events["currency"] == ccy].sort_values("date")
        # Surprise magnitude, expressed in that series' own historical volatility so
        # e.g. a rate release and an industrial-production release are comparable.
        surprise = ccy_events["change"] / ccy_events["actual"].rolling(20, min_periods=5).std().replace(0, np.nan)
        surprise_by_date = surprise.groupby(ccy_events["date"]).mean()
        count_by_date = ccy_events.groupby("date").size()

        surprise_on_dates = surprise_by_date.reindex(unique_dates).sort_index().ffill()
        out[f"{ccy.lower()}_last_surprise_z"] = surprise_on_dates.reindex(dates).to_numpy()

        event_dates = pd.Series(count_by_date.index, index=count_by_date.index)
        last_event_date = event_dates.reindex(unique_dates).sort_index().ffill()
        days_since = (pd.Series(unique_dates, index=unique_dates) - last_event_date).dt.days
        out[f"days_since_{ccy.lower()}_event"] = days_since.reindex(dates).to_numpy()

        is_event_day = count_by_date.reindex(unique_dates).fillna(0).clip(upper=1)
        out[f"is_{ccy.lower()}_event_day"] = is_event_day.reindex(dates).to_numpy()

    return out


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    ind = add_indicators(df)
    close = ind["Close"].astype(float)
    feat = pd.DataFrame(index=ind.index)
    feat["ret_1"] = close.pct_change(1)
    feat["ret_4"] = close.pct_change(4)
    feat["ret_24"] = close.pct_change(24)
    atr = ind["ATR14"].replace(0, np.nan)
    feat["dist_ema20_atr"] = (close - ind["EMA20"]) / atr
    feat["dist_ema50_atr"] = (close - ind["EMA50"]) / atr
    feat["dist_ema200_atr"] = (close - ind["EMA200"]) / atr
    feat["rsi14"] = ind["RSI14"]
    feat["macd_hist"] = ind["MACDHist"]
    feat["bb_width_atr"] = (ind["BBUp"] - ind["BBLow"]) / atr
    feat["bb_position"] = (close - ind["BBLow"]) / (ind["BBUp"] - ind["BBLow"]).replace(0, np.nan)
    feat["atr_pct"] = atr / close
    hour = ind.index.hour
    dow = ind.index.dayofweek
    feat["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    feat["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    feat["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    feat["dow_cos"] = np.cos(2 * np.pi * dow / 7)

    macro = load_macro_features(ind.index)
    feat = feat.join(macro)
    feat["_atr"] = atr
    feat["_close"] = close
    return feat


def triple_barrier_labels(df: pd.DataFrame, atr: pd.Series, horizon: int, mult: float) -> pd.Series:
    """1 = upper (favorable-long) barrier hit first, 0 = lower hit first, NaN = timeout/no barrier hit."""
    high = df["High"].to_numpy(dtype=float)
    low = df["Low"].to_numpy(dtype=float)
    close = df["Close"].to_numpy(dtype=float)
    atr_vals = atr.to_numpy(dtype=float)
    n = len(df)
    labels = np.full(n, np.nan)
    r_multiple = np.full(n, np.nan)
    for i in range(n - horizon):
        if np.isnan(atr_vals[i]) or atr_vals[i] <= 0:
            continue
        entry = close[i]
        upper = entry + mult * atr_vals[i]
        lower = entry - mult * atr_vals[i]
        window_high = high[i + 1 : i + 1 + horizon]
        window_low = low[i + 1 : i + 1 + horizon]
        hit_up = np.argmax(window_high >= upper) if np.any(window_high >= upper) else -1
        hit_down = np.argmax(window_low <= lower) if np.any(window_low <= lower) else -1
        if hit_up == -1 and hit_down == -1:
            continue
        if hit_down == -1 or (hit_up != -1 and hit_up <= hit_down):
            labels[i] = 1.0
            r_multiple[i] = 1.0
        else:
            labels[i] = 0.0
            r_multiple[i] = -1.0
    return pd.Series(labels, index=df.index), pd.Series(r_multiple, index=df.index)


def main() -> None:
    raise SystemExit(
        "This legacy PoC has unsafe publication-date alignment and unpurged labels. "
        "Use: python forex_ml.py --symbols EURUSD GBPUSD USDJPY"
    )


def legacy_main_unsafe() -> None:
    df = load_ohlc(SYMBOL)
    feat = build_features(df)
    labels, r_multiple = triple_barrier_labels(df, feat["_atr"], HORIZON_BARS, BARRIER_ATR_MULT)

    feature_cols = [c for c in feat.columns if not c.startswith("_")]
    data = feat[feature_cols].copy()
    data["label"] = labels
    data["r_multiple"] = r_multiple
    data = data.dropna(subset=["label"])
    data = data.dropna(subset=feature_cols, how="all")

    print(f"Usable rows after labeling: {len(data)} (of {len(feat)} bars)")
    print(f"Base rate (label=1): {data['label'].mean():.4f}")

    split_idx = int(len(data) * (1 - TEST_FRACTION))
    train, test = data.iloc[:split_idx], data.iloc[split_idx:]
    print(f"Train: {train.index.min()} -> {train.index.max()} ({len(train)} rows)")
    print(f"Test:  {test.index.min()} -> {test.index.max()} ({len(test)} rows)")

    X_train, y_train = train[feature_cols], train["label"]
    X_test, y_test = test[feature_cols], test["label"]

    model = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.05,
        max_depth=4,
        l2_regularization=1.0,
        random_state=42,
    )
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)
    print(f"\nTest ROC-AUC: {auc:.4f} (0.50 = no edge)")

    for thr in (0.55, 0.60, 0.65, 0.70):
        mask_long = proba >= thr
        mask_short = proba <= (1 - thr)
        print(f"\n--- Confidence threshold {thr:.2f} ---")
        for name, mask, sign in (("LONG-favored", mask_long, 1.0), ("SHORT-favored", mask_short, -1.0)):
            subset_r = test.loc[mask, "r_multiple"].to_numpy() * sign
            if len(subset_r) < 5:
                print(f"{name}: too few signals ({len(subset_r)})")
                continue
            result = stationary_bootstrap_mean_test(subset_r, simulations=2000, mean_block_length=5.0, seed=42)
            p_adj = bonferroni_adjust(result["p_value"], trials=8)  # 4 thresholds x 2 sides
            print(
                f"{name}: n={len(subset_r)} mean_R={result['observed_mean']:.4f} "
                f"95%CI=[{result['ci_low']:.4f}, {result['ci_high']:.4f}] "
                f"p={result['p_value']:.4f} p_adj={p_adj:.4f}"
            )

    importances = pd.Series(
        _permutation_importance_fallback(model, X_test, y_test), index=feature_cols
    ).sort_values(ascending=False)
    print("\nTop features (permutation importance, test set):")
    print(importances.head(10).to_string())


def _permutation_importance_fallback(model, X_test, y_test):
    from sklearn.inspection import permutation_importance

    result = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=42, scoring="roc_auc")
    return result.importances_mean


if __name__ == "__main__":
    main()
