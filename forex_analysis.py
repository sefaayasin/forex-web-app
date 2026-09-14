"""Market-structure, movement-phase, and directional-bias analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from forex_indicators import add_indicators


@dataclass
class MarketStructureResult:
    ma_direction: str
    structure: str
    combined_direction: str
    phase: str
    response_side: str
    entry_confirmed: bool
    rsi_regime: str
    rsi_divergence: str
    rsi_momentum_break: str
    bb_state: str
    bb_trend_signal: str
    bb_band_walk: str
    bb_pattern: str
    bb_mean_reversion_side: str
    bb_mid_target: Optional[float]
    bb_width_percentile: Optional[float]
    macd_regime: str
    macd_momentum_state: str
    macd_divergence: str
    macd_whipsaw: bool
    macd_atr: Optional[float]
    macd_hist_atr: Optional[float]
    explanation: str


def market_structure_frame(
    df: pd.DataFrame,
    structure_window: int = 8,
    impulse_lookback: int = 12,
    correction_bars: int = 5,
    min_impulse_atr: float = 1.0,
    min_retrace: float = 0.20,
    max_retrace: float = 0.80,
    response_body_atr: float = 0.20,
    divergence_window: int = 8,
    rsi_break_window: int = 8,
    bb_history: int = 200,
) -> pd.DataFrame:
    """MA yönü, swing yapısı ve düzeltme/tepki durumunu ileri bakış olmadan üretir."""
    if df is None or df.empty:
        return pd.DataFrame()

    out = add_indicators(df) if "EMA200" not in df.columns else df.copy()
    close = out["Close"].astype(float)
    open_ = out["Open"].astype(float)
    high = out["High"].astype(float)
    low = out["Low"].astype(float)
    atr = out["ATR14"].astype(float).replace(0, np.nan)
    structure_window = max(int(structure_window), 3)
    impulse_lookback = max(int(impulse_lookback), 4)
    correction_bars = max(int(correction_bars), 2)
    divergence_window = max(int(divergence_window), 4)
    rsi_break_window = max(int(rsi_break_window), 4)

    ema50_slope = out["EMA50"] - out["EMA50"].shift(8)
    ema200_slope = out["EMA200"] - out["EMA200"].shift(20)
    ma_long = (
        (close > out["EMA50"])
        & (out["EMA50"] > out["EMA200"])
        & (ema50_slope > 0)
        & (ema200_slope >= 0)
    )
    ma_short = (
        (close < out["EMA50"])
        & (out["EMA50"] < out["EMA200"])
        & (ema50_slope < 0)
        & (ema200_slope <= 0)
    )

    # Ardışık iki geçmiş pencerenin tepe/diplerini karşılaştırır. Centered rolling
    # kullanılmadığı için gelecekteki mumlar geçmiş yapıyı değiştiremez.
    recent_high = high.rolling(structure_window).max()
    recent_low = low.rolling(structure_window).min()
    prior_high = high.shift(structure_window).rolling(structure_window).max()
    prior_low = low.shift(structure_window).rolling(structure_window).min()
    structure_long = (recent_high > prior_high) & (recent_low > prior_low)
    structure_short = (recent_high < prior_high) & (recent_low < prior_low)

    out["MADirection"] = np.select([ma_long, ma_short], ["BULLISH", "BEARISH"], default="NEUTRAL")
    out["MarketStructure"] = np.select(
        [structure_long, structure_short], ["BULLISH", "BEARISH"], default="RANGE"
    )
    combined_long = ma_long & structure_long
    combined_short = ma_short & structure_short
    out["CombinedDirection"] = np.select(
        [combined_long, combined_short], ["LONG", "SHORT"], default="NONE"
    )

    # İmpuls penceresi düzeltme penceresinden önce biter; böylece düzeltme,
    # impulsun kendi dip/tepe hesabına karışmaz.
    impulse_high = high.shift(correction_bars).rolling(impulse_lookback).max()
    impulse_low = low.shift(correction_bars).rolling(impulse_lookback).min()
    impulse_range = (impulse_high - impulse_low).replace(0, np.nan)
    correction_low = low.shift(1).rolling(correction_bars).min()
    correction_high = high.shift(1).rolling(correction_bars).max()
    long_retrace = (impulse_high - correction_low) / impulse_range
    short_retrace = (correction_high - impulse_low) / impulse_range
    impulse_ok = impulse_range >= (atr * float(min_impulse_atr))

    long_correction = (
        combined_long
        & impulse_ok
        & long_retrace.between(float(min_retrace), float(max_retrace))
        & (correction_low > impulse_low)
    )
    short_correction = (
        combined_short
        & impulse_ok
        & short_retrace.between(float(min_retrace), float(max_retrace))
        & (correction_high < impulse_high)
    )

    body = close - open_
    # Tepki mumu zaten bandın ucuna veya RSI aşırı bölgesine dayanmışsa hareket
    # tükenmiş sayılır; bu durumda giriş, tepkinin peşinden geç kalmış olur.
    band_pos_now = (close - out["BBLow"]) / (out["BBUp"] - out["BBLow"]).replace(0, np.nan)
    long_not_exhausted = (band_pos_now <= 0.85) & (out["RSI14"] <= 72)
    short_not_exhausted = (band_pos_now >= 0.15) & (out["RSI14"] >= 28)
    long_response = (
        long_correction
        & (close > high.shift(1))
        & (body >= atr * float(response_body_atr))
        & (close > out["EMA50"])
        & long_not_exhausted
    )
    short_response = (
        short_correction
        & (close < low.shift(1))
        & (-body >= atr * float(response_body_atr))
        & (close < out["EMA50"])
        & short_not_exhausted
    )
    out["ResponseSide"] = np.select([long_response, short_response], ["LONG", "SHORT"], default="NONE")
    out["CorrectionActive"] = long_correction | short_correction
    out["Retracement"] = np.where(combined_long, long_retrace, np.where(combined_short, short_retrace, np.nan))

    # RSI 50 çevresinde tampon bölge kullanılır; 49/51 gibi küçük geçişler
    # trend değişimi sayılmaz. RSI hiçbir zaman tek başına ters yön üretmez.
    rsi = out["RSI14"].astype(float)
    out["RSIRegime"] = np.select(
        [rsi >= 52.0, rsi <= 48.0], ["BULLISH", "BEARISH"], default="NEUTRAL"
    )

    # Uyumsuzluk, geleceğe bakan centered pivot yerine ardışık iki tamamlanmış
    # pencerenin fiyat ve RSI uçlarını karşılaştırır.
    recent_price_high = high.rolling(divergence_window).max()
    prior_price_high = high.shift(divergence_window).rolling(divergence_window).max()
    recent_price_low = low.rolling(divergence_window).min()
    prior_price_low = low.shift(divergence_window).rolling(divergence_window).min()
    recent_rsi_high = rsi.rolling(divergence_window).max()
    prior_rsi_high = rsi.shift(divergence_window).rolling(divergence_window).max()
    recent_rsi_low = rsi.rolling(divergence_window).min()
    prior_rsi_low = rsi.shift(divergence_window).rolling(divergence_window).min()
    bearish_divergence = (recent_price_high > prior_price_high) & (recent_rsi_high < prior_rsi_high)
    bullish_divergence = (recent_price_low < prior_price_low) & (recent_rsi_low > prior_rsi_low)
    out["RSIDivergence"] = np.select(
        [bearish_divergence, bullish_divergence], ["BEARISH", "BULLISH"], default="NONE"
    )

    # Subjektif trend çizgisi yerine son tamamlanmış RSI aralığının kırılması
    # erken momentum uyarısı olarak raporlanır; tek başına giriş üretmez.
    prior_rsi_ceiling = rsi.shift(1).rolling(rsi_break_window).max()
    prior_rsi_floor = rsi.shift(1).rolling(rsi_break_window).min()
    bullish_rsi_break = (rsi > prior_rsi_ceiling) & (rsi >= 50)
    bearish_rsi_break = (rsi < prior_rsi_floor) & (rsi <= 50)
    out["RSIMomentumBreak"] = np.select(
        [bullish_rsi_break, bearish_rsi_break], ["BULLISH", "BEARISH"], default="NONE"
    )

    # MACD ham büyüklüğü parite fiyatına bağlıdır; ATR ile normalize edilerek
    # semboller/zaman dilimleri arasında daha anlamlı güç ölçümü sağlanır.
    macd = out["MACD"].astype(float)
    macd_signal = out["MACDSignal"].astype(float)
    macd_hist = out["MACDHist"].astype(float)
    out["MACDATR"] = macd / atr
    out["MACDSignalATR"] = macd_signal / atr
    out["MACDHistATR"] = macd_hist / atr
    out["MACDRegime"] = np.select(
        [(macd > 0) & (macd_signal > 0), (macd < 0) & (macd_signal < 0)],
        ["BULLISH", "BEARISH"],
        default="TRANSITION",
    )

    hist_delta = macd_hist.diff()
    bullish_acceleration = (macd_hist > 0) & (hist_delta > 0) & (hist_delta.shift(1) > 0)
    bearish_acceleration = (macd_hist < 0) & (hist_delta < 0) & (hist_delta.shift(1) < 0)
    bullish_weakening = (macd_hist > 0) & (hist_delta < 0) & (hist_delta.shift(1) < 0)
    bearish_weakening = (macd_hist < 0) & (hist_delta > 0) & (hist_delta.shift(1) > 0)
    out["MACDMomentumState"] = np.select(
        [bullish_acceleration, bearish_acceleration, bullish_weakening, bearish_weakening],
        ["BULLISH_ACCELERATION", "BEARISH_ACCELERATION", "BULLISH_WEAKENING", "BEARISH_WEAKENING"],
        default="MIXED",
    )

    hist_sign = np.sign(macd_hist.fillna(0))
    hist_cross = ((hist_sign != hist_sign.shift(1)) & (hist_sign != 0) & (hist_sign.shift(1) != 0)).astype(int)
    cross_count = hist_cross.rolling(12).sum()
    macd_whipsaw = (
        (cross_count >= 4)
        & (out["MACDATR"].abs() <= 0.20)
        & (out["MACDHistATR"].abs() <= 0.08)
    )
    out["MACDWhipsaw"] = macd_whipsaw

    recent_macd_high = macd.rolling(divergence_window).max()
    prior_macd_high = macd.shift(divergence_window).rolling(divergence_window).max()
    recent_macd_low = macd.rolling(divergence_window).min()
    prior_macd_low = macd.shift(divergence_window).rolling(divergence_window).min()
    recent_hist_high = macd_hist.rolling(divergence_window).max()
    prior_hist_high = macd_hist.shift(divergence_window).rolling(divergence_window).max()
    recent_hist_low = macd_hist.rolling(divergence_window).min()
    prior_hist_low = macd_hist.shift(divergence_window).rolling(divergence_window).min()
    bearish_macd_divergence = (recent_price_high > prior_price_high) & (
        (recent_macd_high < prior_macd_high) | (recent_hist_high < prior_hist_high)
    )
    bullish_macd_divergence = (recent_price_low < prior_price_low) & (
        (recent_macd_low > prior_macd_low) | (recent_hist_low > prior_hist_low)
    )
    out["MACDDivergence"] = np.select(
        [bearish_macd_divergence, bullish_macd_divergence],
        ["BEARISH", "BULLISH"],
        default="NONE",
    )

    # Bollinger rejimi: mutlak genişlik yerine sembolün kendi yakın tarihine
    # göre yüzdelik kullanılır. Daralma yön tahmini değildir; kırılım ayrıca teyit edilir.
    bb_mid = out["BBMid"].astype(float)
    bb_up = out["BBUp"].astype(float)
    bb_low = out["BBLow"].astype(float)
    bb_width = ((bb_up - bb_low) / bb_mid.abs().replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    bb_history = max(int(bb_history), 60)
    min_bb_history = max(40, bb_history // 3)
    bb_q10 = bb_width.shift(1).rolling(bb_history, min_periods=min_bb_history).quantile(0.10)
    bb_q25 = bb_width.shift(1).rolling(bb_history, min_periods=min_bb_history).quantile(0.25)
    bb_q95 = bb_width.shift(1).rolling(bb_history, min_periods=min_bb_history).quantile(0.95)
    out["BBWidth"] = bb_width
    out["BBWidthPercentile"] = bb_width.rolling(bb_history, min_periods=min_bb_history).apply(
        lambda values: float(np.mean(values <= values[-1])), raw=True
    )
    ema_gap_atr = (out["EMA20"] - out["EMA50"]).abs() / atr
    ema50_slope_atr = (out["EMA50"] - out["EMA50"].shift(20)).abs() / atr
    bb_width_atr = (bb_up - bb_low).abs() / atr
    trend_regime = (ema50_slope_atr >= 0.80) & (ema_gap_atr >= 0.30)
    range_regime = (ema50_slope_atr <= 0.60) & (ema_gap_atr <= 0.40) & (bb_width_atr <= 4.0)
    out["StrategyRegime"] = np.select(
        [trend_regime, range_regime], ["TREND", "RANGE"], default="TRANSITION"
    )
    out["RegimeEMAGapATR"] = ema_gap_atr
    out["RegimeSlopeATR"] = ema50_slope_atr
    out["RegimeBBWidthATR"] = bb_width_atr
    extreme_squeeze = bb_width <= bb_q10
    standard_contraction = (bb_width <= bb_q25) & (bb_width < bb_width.shift(1))
    extreme_expansion = bb_width >= bb_q95
    width_expanding = (bb_width > bb_width.shift(1)) & (bb_width.shift(1) >= bb_width.shift(2))
    compressed_recently = (extreme_squeeze | standard_contraction).shift(1).rolling(6).max().fillna(0).astype(bool)

    prior_swing_high = high.shift(1).rolling(structure_window).max()
    prior_swing_low = low.shift(1).rolling(structure_window).min()
    strong_bull_candle = (body >= atr * 0.50) & (close > bb_up)
    strong_bear_candle = (-body >= atr * 0.50) & (close < bb_low)
    bullish_expansion = (
        compressed_recently & width_expanding & strong_bull_candle
        & (close > prior_swing_high) & combined_long & (out["RSIRegime"] == "BULLISH")
    )
    bearish_expansion = (
        compressed_recently & width_expanding & strong_bear_candle
        & (close < prior_swing_low) & combined_short & (out["RSIRegime"] == "BEARISH")
    )
    out["BBTrendSignal"] = np.select(
        [bullish_expansion, bearish_expansion], ["LONG", "SHORT"], default="NONE"
    )

    band_pos = ((close - bb_low) / (bb_up - bb_low).replace(0, np.nan)).clip(-1, 2)
    upper_walk_count = (band_pos >= 0.90).rolling(5).sum()
    lower_walk_count = (band_pos <= 0.10).rolling(5).sum()
    bullish_band_walk = (upper_walk_count >= 3) & (bb_mid > bb_mid.shift(5)) & combined_long
    bearish_band_walk = (lower_walk_count >= 3) & (bb_mid < bb_mid.shift(5)) & combined_short
    out["BBBandWalk"] = np.select(
        [bullish_band_walk, bearish_band_walk], ["BULLISH", "BEARISH"], default="NONE"
    )

    # Teyitli W/M: eski pencerede bant ihlali, ikinci testte bandın içinde kalma
    # ve son olarak aradaki swing seviyesinin kapanışla kırılması gerekir.
    pattern_window = 6
    old_lower_breach = (low.shift(pattern_window + 1) < bb_low.shift(pattern_window + 1)).rolling(pattern_window).max().fillna(0).astype(bool)
    old_upper_breach = (high.shift(pattern_window + 1) > bb_up.shift(pattern_window + 1)).rolling(pattern_window).max().fillna(0).astype(bool)
    recent_lower_breach = (low.shift(1) < bb_low.shift(1)).rolling(pattern_window).max().fillna(0).astype(bool)
    recent_upper_breach = (high.shift(1) > bb_up.shift(1)).rolling(pattern_window).max().fillna(0).astype(bool)
    first_low = low.shift(pattern_window + 1).rolling(pattern_window).min()
    first_high = high.shift(pattern_window + 1).rolling(pattern_window).max()
    second_low = low.shift(1).rolling(pattern_window).min()
    second_high = high.shift(1).rolling(pattern_window).max()
    neckline_high = high.shift(1).rolling(pattern_window).max()
    neckline_low = low.shift(1).rolling(pattern_window).min()
    flat_middle = (bb_mid - bb_mid.shift(8)).abs() <= atr * 0.50
    w_confirmed = (
        old_lower_breach & ~recent_lower_breach & (second_low <= first_low + atr * 0.50)
        & (close > neckline_high) & (body >= atr * 0.20)
    )
    m_confirmed = (
        old_upper_breach & ~recent_upper_breach & (second_high >= first_high - atr * 0.50)
        & (close < neckline_low) & (-body >= atr * 0.20)
    )
    out["BBPattern"] = np.select([w_confirmed, m_confirmed], ["W_CONFIRMED", "M_CONFIRMED"], default="NONE")

    # Mean-reversion yalnız yatay orta bantta ve en az 1R alan varsa adaydır;
    # ana trend motoruna otomatik ters işlem göndermez.
    long_reentry = (close.shift(1) < bb_low.shift(1)) & (close > bb_low) & flat_middle
    short_reentry = (close.shift(1) > bb_up.shift(1)) & (close < bb_up) & flat_middle
    long_stop_distance = (close - (low.rolling(4).min() - atr * 0.15)).clip(lower=np.finfo(float).eps)
    short_stop_distance = ((high.rolling(4).max() + atr * 0.15) - close).clip(lower=np.finfo(float).eps)
    long_mid_reward = bb_mid - close
    short_mid_reward = close - bb_mid
    long_mean_reversion = (
        (long_reentry | w_confirmed) & (out["RSIRegime"] != "BEARISH")
        & (long_mid_reward > 0) & (long_mid_reward / long_stop_distance >= 1.0)
    )
    short_mean_reversion = (
        (short_reentry | m_confirmed) & (out["RSIRegime"] != "BULLISH")
        & (short_mid_reward > 0) & (short_mid_reward / short_stop_distance >= 1.0)
    )
    out["BBMeanReversionSide"] = np.select(
        [long_mean_reversion, short_mean_reversion], ["LONG", "SHORT"], default="NONE"
    )
    # Ayrı RANGE motoru: bir önceki mum bandı ihlal/test eder, kapanmış
    # mevcut mum yeniden bandın içine momentum dönüşüyle girer. Bu sinyal
    # trend motoruna gönderilmez ve hedefi yalnız orta banttır.
    prior_lower_test = (low.shift(1) <= bb_low.shift(1)) | (close.shift(1) < bb_low.shift(1))
    prior_upper_test = (high.shift(1) >= bb_up.shift(1)) | (close.shift(1) > bb_up.shift(1))
    range_long_reversal = (
        range_regime & prior_lower_test & (close > bb_low) & (body >= atr * 0.10)
        & (rsi > rsi.shift(1)) & (rsi <= 55) & (bb_mid > close)
    )
    range_short_reversal = (
        range_regime & prior_upper_test & (close < bb_up) & (-body >= atr * 0.10)
        & (rsi < rsi.shift(1)) & (rsi >= 45) & (bb_mid < close)
    )
    out["RangeReversalSignal"] = np.select(
        [range_long_reversal, range_short_reversal], ["LONG", "SHORT"], default="NONE"
    )
    out["BBMidTarget"] = bb_mid
    out["BBExtremeVolatility"] = extreme_expansion
    out["BBState"] = np.select(
        [bullish_expansion, bearish_expansion, bullish_band_walk, bearish_band_walk,
         extreme_expansion, extreme_squeeze, standard_contraction],
        ["BULLISH VOLATİLİTE AÇILIMI", "BEARISH VOLATİLİTE AÇILIMI",
         "BULLISH BAND WALK", "BEARISH BAND WALK", "AŞIRI GENİŞ / HABER RİSKİ",
         "AŞIRI SIKIŞMA – YÖN BEKLENİYOR", "STANDART DARALMA – HAZIRLIK"],
        default="NORMAL VOLATİLİTE",
    )
    return out


def evaluate_market_structure(df: pd.DataFrame) -> MarketStructureResult:
    if df is None or df.empty or len(df) < 60:
        return MarketStructureResult(
            "NEUTRAL", "RANGE", "NONE", "VERİ YETERSİZ", "NONE", False,
            "NEUTRAL", "NONE", "NONE", "VERİ YETERSİZ", "NONE", "NONE", "NONE",
            "NONE", None, None, "TRANSITION", "MIXED", "NONE", False, None, None,
            "Yeterli mum yok",
        )
    model = market_structure_frame(df)
    valid = model.dropna(subset=["EMA50", "EMA200", "ATR14"])
    if valid.empty:
        return MarketStructureResult(
            "NEUTRAL", "RANGE", "NONE", "VERİ YETERSİZ", "NONE", False,
            "NEUTRAL", "NONE", "NONE", "VERİ YETERSİZ", "NONE", "NONE", "NONE",
            "NONE", None, None, "TRANSITION", "MIXED", "NONE", False, None, None,
            "Göstergeler hazır değil",
        )
    row = valid.iloc[-1]
    ma_direction = str(row["MADirection"])
    structure = str(row["MarketStructure"])
    combined = str(row["CombinedDirection"])
    response = str(row["ResponseSide"])
    correction = bool(row["CorrectionActive"])
    rsi_regime = str(row["RSIRegime"])
    rsi_divergence = str(row["RSIDivergence"])
    rsi_momentum_break = str(row["RSIMomentumBreak"])
    bb_state = str(row["BBState"])
    bb_trend_signal = str(row["BBTrendSignal"])
    bb_band_walk = str(row["BBBandWalk"])
    bb_pattern = str(row["BBPattern"])
    bb_mean_reversion_side = str(row["BBMeanReversionSide"])
    bb_mid_target = None if pd.isna(row["BBMidTarget"]) else float(row["BBMidTarget"])
    bb_width_percentile = None if pd.isna(row["BBWidthPercentile"]) else float(row["BBWidthPercentile"])
    macd_regime = str(row["MACDRegime"])
    macd_momentum_state = str(row["MACDMomentumState"])
    macd_divergence = str(row["MACDDivergence"])
    macd_whipsaw_value = bool(row["MACDWhipsaw"])
    macd_atr_value = None if pd.isna(row["MACDATR"]) else float(row["MACDATR"])
    macd_hist_atr_value = None if pd.isna(row["MACDHistATR"]) else float(row["MACDHistATR"])
    if response in {"LONG", "SHORT"}:
        phase = "TEPKİ TEYİTLİ"
    elif correction:
        phase = "DÜZELTME"
    elif combined in {"LONG", "SHORT"}:
        phase = "İMPULS / TREND"
    else:
        phase = "UYUMSUZ / YATAY"
    retracement = row.get("Retracement", np.nan)
    retrace_text = "-" if pd.isna(retracement) else f"%{float(retracement) * 100:.0f}"
    explanation = (
        f"MA: {ma_direction}; yapı: {structure}; faz: {phase}; düzeltme: {retrace_text}; "
        f"RSI rejimi: {rsi_regime}; uyumsuzluk: {rsi_divergence}; momentum kırılımı: {rsi_momentum_break}."
        f" Bollinger: {bb_state}; trend sinyali: {bb_trend_signal}; formasyon: {bb_pattern}."
        f" MACD rejimi: {macd_regime}; histogram: {macd_momentum_state}; "
        f"uyumsuzluk: {macd_divergence}; whipsaw: {'EVET' if macd_whipsaw_value else 'HAYIR'}."
    )
    return MarketStructureResult(
        ma_direction=ma_direction,
        structure=structure,
        combined_direction=combined,
        phase=phase,
        response_side=response,
        entry_confirmed=response == combined and response in {"LONG", "SHORT"},
        rsi_regime=rsi_regime,
        rsi_divergence=rsi_divergence,
        rsi_momentum_break=rsi_momentum_break,
        bb_state=bb_state,
        bb_trend_signal=bb_trend_signal,
        bb_band_walk=bb_band_walk,
        bb_pattern=bb_pattern,
        bb_mean_reversion_side=bb_mean_reversion_side,
        bb_mid_target=bb_mid_target,
        bb_width_percentile=bb_width_percentile,
        macd_regime=macd_regime,
        macd_momentum_state=macd_momentum_state,
        macd_divergence=macd_divergence,
        macd_whipsaw=macd_whipsaw_value,
        macd_atr=macd_atr_value,
        macd_hist_atr=macd_hist_atr_value,
        explanation=explanation,
    )


def calculate_stop_target_distances(
    history: pd.DataFrame,
    entry: float,
    side: str,
    atr: float,
    atr_mult: float,
    rr: float,
    stop_mode: str = "ATR",
    target_mode: str = "Sabit R",
    swing_lookback: int = 10,
) -> tuple[float, float]:
    """Canlı ve backtest tarafından paylaşılan ATR/yapısal stop-hedef hesabı."""
    atr_distance = max(float(atr) * float(atr_mult), np.finfo(float).eps)
    recent = history.tail(max(int(swing_lookback), 2)) if history is not None else pd.DataFrame()
    structural_distance = atr_distance
    if not recent.empty:
        if side == "LONG":
            swing = float(recent["Low"].min())
            structural_distance = max(entry - (swing - atr * 0.15), atr * 0.5)
        else:
            swing = float(recent["High"].max())
            structural_distance = max((swing + atr * 0.15) - entry, atr * 0.5)
    if stop_mode == "Swing + ATR":
        stop_distance = structural_distance
    elif stop_mode == "Hibrit (uzak olan)":
        stop_distance = max(atr_distance, structural_distance)
    else:
        stop_distance = atr_distance

    target_distance = stop_distance * float(rr)
    if target_mode == "Yapı / minimum 1R" and not recent.empty:
        opposite_range = (
            float(recent["High"].max()) - entry
            if side == "LONG"
            else entry - float(recent["Low"].min())
        )
        target_distance = max(stop_distance, opposite_range)
    return float(stop_distance), float(target_distance)

# =============================================================================
# DECISION ENGINE
# =============================================================================

@dataclass
class BiasResult:
    label: str
    score: float
    trend_score: float
    momentum_score: float
    volatility_score: float
    explanation: str


def label_from_score(score: float) -> str:
    if score >= 65:
        return "Güçlü Alım Yönlü"
    if score >= 25:
        return "Alım Yönlü"
    if score <= -65:
        return "Güçlü Satış Yönlü"
    if score <= -25:
        return "Satış Yönlü"
    return "İşlem Yok"


def latest_valid_row(df: pd.DataFrame) -> Optional[pd.Series]:
    if df.empty:
        return None
    needed = ["Close", "EMA20", "EMA50", "EMA200", "RSI14", "MACD", "MACDSignal", "MACDHist", "BBLow", "BBMid", "BBUp", "ATR14"]
    valid = df.dropna(subset=[c for c in needed if c in df.columns])
    if valid.empty:
        return None
    return valid.iloc[-1]


def evaluate_bias(df: pd.DataFrame) -> BiasResult:
    """Tek bar değil, trend + momentum + volatilite yapısına göre bias üretir."""
    if df.empty or len(df) < 60:
        return BiasResult("İşlem Yok", 0, 0, 0, 0, "Yeterli veri yok")

    ind = add_indicators(df)
    row = latest_valid_row(ind)
    if row is None:
        return BiasResult("İşlem Yok", 0, 0, 0, 0, "İndikatörler için yeterli veri yok")

    close = float(row["Close"])
    ema20 = float(row["EMA20"])
    ema50 = float(row["EMA50"])
    ema200 = float(row["EMA200"])
    rsi = float(row["RSI14"])
    macd = float(row["MACD"])
    sig = float(row["MACDSignal"])
    hist = float(row["MACDHist"])
    bb_low = float(row["BBLow"])
    bb_mid = float(row["BBMid"])
    bb_up = float(row["BBUp"])
    atr = float(row["ATR14"])

    # EMA eğimleri
    ema50_slope = float(ind["EMA50"].iloc[-1] - ind["EMA50"].iloc[-8]) if len(ind) > 8 else 0.0
    ema200_slope = float(ind["EMA200"].iloc[-1] - ind["EMA200"].iloc[-20]) if len(ind) > 20 else 0.0
    hist_delta = float(ind["MACDHist"].iloc[-1] - ind["MACDHist"].iloc[-4]) if len(ind) > 4 else 0.0

    trend_score = 0.0
    reasons: list[str] = []

    # Trend skoru
    if close > ema200:
        trend_score += 20
        reasons.append("Fiyat EMA200 üzerinde")
    else:
        trend_score -= 20
        reasons.append("Fiyat EMA200 altında")

    if ema50 > ema200:
        trend_score += 20
        reasons.append("EMA50 > EMA200")
    else:
        trend_score -= 20
        reasons.append("EMA50 < EMA200")

    if close > ema20 > ema50:
        trend_score += 15
        reasons.append("Kısa vadeli trend yukarı")
    elif close < ema20 < ema50:
        trend_score -= 15
        reasons.append("Kısa vadeli trend aşağı")

    if ema50_slope > 0 and ema200_slope >= 0:
        trend_score += 10
        reasons.append("EMA eğimleri pozitif")
    elif ema50_slope < 0 and ema200_slope <= 0:
        trend_score -= 10
        reasons.append("EMA eğimleri negatif")

    # Momentum skoru
    momentum_score = 0.0
    if 52 <= rsi <= 68:
        momentum_score += 20
        reasons.append("RSI alım momentumunda")
    elif rsi > 75:
        momentum_score += 5
        reasons.append("RSI çok yüksek, momentum var ama geri çekilme riski yüksek")
    elif 32 <= rsi <= 48:
        momentum_score -= 20
        reasons.append("RSI satış momentumunda")
    elif rsi < 25:
        momentum_score -= 5
        reasons.append("RSI çok düşük, satış baskısı var ama tepki riski yüksek")

    recent_hist = ind["MACDHist"].dropna().tail(13)
    recent_sign = np.sign(recent_hist)
    cross_count = int(((recent_sign != recent_sign.shift(1)) & (recent_sign != 0) & (recent_sign.shift(1) != 0)).sum())
    macd_atr = macd / atr if atr > 0 else 0.0
    hist_atr = hist / atr if atr > 0 else 0.0
    macd_whipsaw = cross_count >= 4 and abs(macd_atr) <= 0.20 and abs(hist_atr) <= 0.08
    if macd_whipsaw:
        reasons.append("MACD sıfır çevresinde whipsaw; kesişim puanı yok")
    else:
        if macd > 0 and sig > 0:
            momentum_score += 10
            reasons.append("MACD sıfır üstü bullish rejimde")
        elif macd < 0 and sig < 0:
            momentum_score -= 10
            reasons.append("MACD sıfır altı bearish rejimde")

        if macd > sig and hist > 0:
            momentum_score += 10
            reasons.append("MACD/sinyal bullish uyumlu")
        elif macd < sig and hist < 0:
            momentum_score -= 10
            reasons.append("MACD/sinyal bearish uyumlu")

        hist_delta_now = float(ind["MACDHist"].iloc[-1] - ind["MACDHist"].iloc[-2])
        hist_delta_prev = float(ind["MACDHist"].iloc[-2] - ind["MACDHist"].iloc[-3])
        if hist > 0 and hist_delta_now > 0 and hist_delta_prev > 0:
            momentum_score += 6
            reasons.append("MACD histogram bullish hızlanıyor")
        elif hist < 0 and hist_delta_now < 0 and hist_delta_prev < 0:
            momentum_score -= 6
            reasons.append("MACD histogram bearish hızlanıyor")
        elif (hist > 0 and hist_delta_now < 0 and hist_delta_prev < 0) or (hist < 0 and hist_delta_now > 0 and hist_delta_prev > 0):
            reasons.append("MACD histogram yavaşlıyor; ters sinyal değil")

    # Bollinger/volatilite skoru: tek başına LONG/SHORT değil, pozisyon kalitesi filtresi.
    volatility_score = 0.0
    if bb_up > bb_low:
        band_pos = (close - bb_low) / (bb_up - bb_low)
        if close > bb_mid and 0.45 <= band_pos <= 0.90:
            volatility_score += 10
            reasons.append("Fiyat BB orta band üstünde, üst banda aşırı yapışmamış")
        elif close < bb_mid and 0.10 <= band_pos <= 0.55:
            volatility_score -= 10
            reasons.append("Fiyat BB orta band altında, alt banda aşırı yapışmamış")
        elif band_pos > 0.95:
            volatility_score += 3
            reasons.append("Fiyat üst banda çok yakın, alımda takip riski var")
        elif band_pos < 0.05:
            volatility_score -= 3
            reasons.append("Fiyat alt banda çok yakın, satışta takip riski var")

    raw_score = trend_score + momentum_score + volatility_score
    score = float(np.clip(raw_score, -100, 100))
    label = label_from_score(score)
    explanation = "; ".join(reasons[:5])
    return BiasResult(label, score, trend_score, momentum_score, volatility_score, explanation)
