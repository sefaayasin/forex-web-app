# Streamlit Forex Analyzer (Web) – 5‑kademeli sinyal + canlı panel
# ===============================================================
# Özellikler
# - Zaman dilimi seçimi: 4 Saat, 1 Saat, 15 Dakika, 5 Dakika
# - Bollinger + RSI + MACD grafikleri (Plotly, 3 panel)
# - Sağ/üst kutuda canlı fiyat, başlangıca göre % değişim
# - Gauge (5 dilim): Kesin Al | Al | Bekle | Sat | Kesin Sat
# - Sembol seçimi (favori listeden), arama/elle yazma
# - Çoklu zaman dilimi analiz tablosu + karar tablosu
# - Canlı Ichimoku-EMA200-RSI tetikleyici paneli (1m), Entry/SL/TP gösterimi
#
# Çalıştırma:
#   pip install streamlit yfinance pandas numpy plotly pytz
#   streamlit run forex_web_app_streamlit.py
#
# Notlar:
# - Veri: Yahoo Finance (yfinance). Bazı egzotik semboller veri döndürmeyebilir.
# - Otomatik yenileme: cache TTL ile 60 sn; “Yenile” butonu ile manuel tetiklenebilir.
# - Saat dilimi: Europe/Istanbul

import time
from datetime import datetime
import pytz
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

TR_TZ = pytz.timezone("Europe/Istanbul")

# ----------------------------- UI CONFIG -----------------------------
st.set_page_config(page_title="Forex Analyzer Web", layout="wide")
st.markdown(
    """
    <style>
    .small-muted { color:#6c757d; font-size:0.9rem; }
    .pill { display:inline-block; padding:4px 10px; border-radius:999px; background:#f1f3f5; margin-right:6px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------- SYMBOLS ------------------------------
SYMBOL_LIST = [
    # Majors
    "EURUSD=X","USDJPY=X","GBPUSD=X","AUDUSD=X","USDCAD=X","USDCHF=X","NZDUSD=X",
    # EUR crosses
    "EURGBP=X","EURAUD=X","EURCAD=X","EURCHF=X","EURJPY=X","EURNZD=X",
    # GBP crosses
    "GBPJPY=X","GBPAUD=X","GBPCAD=X","GBPCHF=X","GBPNZD=X",
    # AUD crosses
    "AUDCAD=X","AUDCHF=X","AUDJPY=X","AUDNZD=X",
    # CAD/CHF/JPY crosses
    "CADCHF=X","CADJPY=X","CHFJPY=X",
    # NZD crosses
    "NZDCAD=X","NZDCHF=X","NZDJPY=X",
    # Exotics (sample)
    "EURZAR=X"
]

# --------------------------- INDICATORS -----------------------------

def get_pip_value(symbol: str) -> float:
    s = symbol.upper().replace("=X", "")
    if len(s) >= 6:
        base, quote = s[:-3], s[-3:]
    else:
        parts = s.replace("-", "").replace("/", "")
        base, quote = parts[:3], parts[-3:]
    if quote == "JPY":
        return 0.01
    if base in {"XAU","XAG"}:
        return 0.1
    if base in {"BTC","ETH"} or quote in {"BTC","ETH"}:
        return 1.0
    return 0.0001


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    close = close.astype(float)
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    roll_up   = gain.rolling(period, min_periods=period).mean()
    roll_down = loss.rolling(period, min_periods=period).mean()
    rs  = roll_up / roll_down
    rsi = 100 - (100 / (1 + rs))
    return rsi.dropna()


def compute_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd  = ema_fast - ema_slow
    sig   = macd.ewm(span=signal, adjust=False).mean()
    hist  = macd - sig
    return macd, sig, hist


def compute_bbands(close: pd.Series, period: int = 20, mult: float = 2.0):
    mid = close.rolling(period).mean()
    std = close.rolling(period).std(ddof=0)
    up  = mid + mult * std
    low = mid - mult * std
    return low, mid, up


# --------------------------- DATA FETCH ------------------------------
@st.cache_data(ttl=60)
def fetch_ohlc(symbol: str, interval: str, period: str) -> pd.DataFrame:
    if interval.lower() in {"4h","4hr","4hour"}:
        base = yf.download(symbol, interval="60m", period="30d", progress=False, auto_adjust=False)
        if base is None or base.empty:
            return pd.DataFrame()
        base = _fix_cols(base)
        out = (
            base.resample("4H").agg({"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"}).dropna()
        )
        return out
    df = yf.download(symbol, interval=interval, period=period, progress=False, auto_adjust=False)
    if df is None or df.empty:
        return pd.DataFrame()
    return _fix_cols(df)


def _fix_cols(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].title() for c in df.columns]
    else:
        df.columns = [str(c).title() for c in df.columns]
    return df


@st.cache_data(ttl=30)
def fetch_last_price(symbol: str) -> float | None:
    try:
        data = yf.download(symbol, period="1d", interval="1m", progress=False, auto_adjust=False)
        if data is None or data.empty:
            return None
        return float(data["Close"].iloc[-1])
    except Exception:
        return None


# --------------------------- DECISIONS -------------------------------
DECISION_SCORES = {
    "Kesin Al" :  2,
    "Al"       :  1,
    "Bekle"    :  0,
    "Sat"      : -1,
    "Kesin Sat": -2
}


def rsi_decision(rsi_series: pd.Series, lookback: int = 3) -> str:
    if len(rsi_series) <= lookback:
        return "Bekle"
    rsi   = rsi_series.iloc[-1]
    slope = rsi - rsi_series.iloc[-lookback]
    if rsi >= 65 and slope > 0: return "Kesin Al"
    if rsi >= 55 and slope > 0: return "Al"
    if rsi <= 35 and slope < 0: return "Kesin Sat"
    if rsi <= 45 and slope < 0: return "Sat"
    return "Bekle"


def macd_decision(macd: pd.Series, sig: pd.Series, hist: pd.Series) -> str:
    if len(hist) < 5:
        return "Bekle"
    macd_now, sig_now, hist_now = macd.iloc[-1], sig.iloc[-1], hist.iloc[-1]
    ref = hist.iloc[-5:].abs().mean()
    ref = max(ref, 1e-9)
    if macd_now > sig_now and hist_now > 0:
        return "Kesin Al" if hist_now > 1.2 * ref else "Al"
    if macd_now < sig_now and hist_now < 0:
        return "Kesin Sat" if abs(hist_now) > 1.2 * ref else "Sat"
    return "Bekle"


def bb_decision(close: pd.Series, low: pd.Series, mid: pd.Series, up: pd.Series, tol: float = 0.995) -> str:
    if len(close) < 5 or len(mid) < 5:
        return "Bekle"
    c_now, l_now, m_now, u_now = close.iloc[-1], low.iloc[-1], mid.iloc[-1], up.iloc[-1]
    mid_slope = m_now - mid.iloc[-4]
    c_prev, m_prev = close.iloc[-2], mid.iloc[-2]
    crossed_up   = (c_prev <= m_prev) and (c_now > m_now)
    crossed_down = (c_prev >= m_prev) and (c_now < m_now)
    if c_now >= u_now: return "Kesin Al"
    if (c_now >= u_now * tol) or (c_now > m_now and mid_slope > 0) or crossed_up: return "Al"
    if c_now <= l_now: return "Kesin Sat"
    if (c_now <= l_now / tol) or (c_now < m_now and mid_slope < 0) or crossed_down: return "Sat"
    return "Bekle"


def overall(votes: list[str]) -> str:
    score = sum(DECISION_SCORES[v] for v in votes)
    if score >= 4: return "Kesin Al"
    if score >= 2: return "Al"
    if score <= -4: return "Kesin Sat"
    if score <= -2: return "Sat"
    return "Bekle"


def analyse_symbol(symbol: str, bb_tol: float = 0.995):
    frames = {
        "4 Saat"    : {"interval":"4h",  "period":"30d"},
        "1 Saat"    : {"interval":"60m", "period":"7d"},
        "15 Dakika" : {"interval":"15m", "period":"5d"},
        "5 Dakika"  : {"interval":"5m",  "period":"1d"},
    }
    tech_rows, dec_rows = [], []
    for tf, prm in frames.items():
        df = fetch_ohlc(symbol, prm["interval"], prm["period"])
        if df.empty or len(df) < 35:
            continue
        df = df.iloc[:-1]  # son barı alma
        close = df["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:,0]
        rsi_s = compute_rsi(close)
        macd_s, sig_s, h = compute_macd(close)
        low_s, mid_s, up_s = compute_bbands(close)
        r_dec = rsi_decision(rsi_s)
        m_dec = macd_decision(macd_s, sig_s, h)
        b_dec = bb_decision(close, low_s, mid_s, up_s, bb_tol)
        fin = overall([r_dec, m_dec, b_dec])
        slope_rsi = rsi_s.iloc[-1] - rsi_s.iloc[-4] if len(rsi_s) > 4 else 0
        r_txt = f"RSI {rsi_s.iloc[-1]:.2f} / eğim: {slope_rsi:+.2f}"
        h_val = h.iloc[-1]; h_prev = h.iloc[-2] if len(h) > 1 else h_val
        h_delta = h_val - h_prev
        m_txt = f"MACD {macd_s.iloc[-1]:.4f}, Sig {sig_s.iloc[-1]:.4f}, Hist {h_val:+.4f}, HistΔ {h_delta:+.4f}"
        mid_delta = mid_s.iloc[-1] - mid_s.iloc[-4] if len(mid_s) > 4 else 0
        pos = ("Üst" if close.iloc[-1] >= up_s.iloc[-1]*bb_tol else
               "Alt" if close.iloc[-1] <= low_s.iloc[-1]/bb_tol else "Orta")
        b_txt = f"{pos} banda yakın; MidΔ {mid_delta:+.5f}"
        tech_rows.append([tf, r_txt, m_txt, b_txt])
        dec_rows.append([tf, r_dec, m_dec, b_dec, fin])
    tech_df = pd.DataFrame(tech_rows, columns=["Zaman Dilimi","RSI","MACD","Bollinger Bands"])
    dec_df  = pd.DataFrame(dec_rows , columns=["Zaman Dilimi","RSI","MACD","Bollinger Bands","Genel Karar"])
    return tech_df, dec_df


def final_global_decision(df_dec: pd.DataFrame) -> str:
    weights = {"4 Saat":4, "1 Saat":3, "15 Dakika":2, "5 Dakika":1}
    score = 0
    for _, row in df_dec.iterrows():
        w = weights.get(row["Zaman Dilimi"], 1)
        score += DECISION_SCORES[row["Genel Karar"]] * w
    if score >= 8: return "Kesin Al"
    if score >= 4: return "Al"
    if score <= -8: return "Kesin Sat"
    if score <= -4: return "Sat"
    return "Bekle"


# ------------------------------ PLOTS -------------------------------

def plot_main_figure(symbol: str, interval: str, period: str):
    df = fetch_ohlc(symbol, interval, period)
    if df.empty:
        return go.Figure(), df
    # TZ
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC").tz_convert(TR_TZ)
    else:
        df.index = df.index.tz_convert(TR_TZ)
    close = df["Close"].astype(float)
    low, mid, up = compute_bbands(close)
    rsi = compute_rsi(close)
    macd, sig, hist = compute_macd(close)

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        row_heights=[0.55, 0.225, 0.225], vertical_spacing=0.04,
                        specs=[[{"secondary_y": False}], [{"secondary_y": False}], [{"secondary_y": False}]])

    # Price + BB
    fig.add_trace(go.Scatter(x=df.index, y=close, name="Fiyat", line=dict(width=1.6)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=up, name="BB Üst", line=dict(width=1, color="rgba(0,102,255,0.4)")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=low, name="BB Alt", fill="tonexty", mode="lines",
                             line=dict(width=1, color="rgba(0,102,255,0.4)")), row=1, col=1)
    ema20 = close.ewm(span=20).mean()
    fig.add_trace(go.Scatter(x=df.index, y=ema20, name="EMA 20", line=dict(width=1, color="#505050")), row=1, col=1)

    # RSI
    fig.add_trace(go.Scatter(x=rsi.index, y=rsi, name="RSI", line=dict(width=1.2, color="#9b59b6")), row=2, col=1)
    fig.add_hline(y=70, line_width=1, line_dash="dash", line_color="#d35400", row=2, col=1)
    fig.add_hline(y=30, line_width=1, line_dash="dash", line_color="#27ae60", row=2, col=1)
    fig.update_yaxes(range=[0,100], row=2, col=1)

    # MACD
    colors = np.where(hist > 0, "#2ecc71", "#e74c3c").tolist()
    fig.add_trace(go.Bar(x=hist.index, y=hist, name="Hist", marker_color=colors), row=3, col=1)
    fig.add_trace(go.Scatter(x=macd.index, y=macd, name="MACD", line=dict(width=1.1, color="#2980b9")), row=3, col=1)
    fig.add_trace(go.Scatter(x=sig.index,  y=sig,  name="Signal", line=dict(width=1.1, color="#f1c40f")), row=3, col=1)

    fig.update_layout(margin=dict(l=30,r=20,t=40,b=30), height=700,
                      title=f"{symbol} | {interval}")
    return fig, df


def gauge_figure(decision: str) -> go.Figure:
    labels = ["Kesin Al","Al","Bekle","Sat","Kesin Sat"]
    colors = ["#1e7e34","#28a745","#6c757d","#dc3545","#bd2130"]
    # yatay 5 bölmeli bar + iğne
    fig = go.Figure()
    x0 = 0
    for i, (lab, col) in enumerate(zip(labels, colors)):
        fig.add_shape(type="rect", x0=i, x1=i+1, y0=0, y1=1, fillcolor=col, line=dict(width=1, color="white"))
        fig.add_annotation(x=i+0.5, y=0.5, text=lab, showarrow=False, font=dict(color="white"))
    idx = labels.index(decision) if decision in labels else 2
    fig.add_shape(type="path",
                  path=f"M {idx+0.5} 1.05 L {idx+0.35} 1.25 L {idx+0.65} 1.25 Z",
                  fillcolor="black", line_color="black")
    fig.update_xaxes(range=[0,5], showgrid=False, visible=False)
    fig.update_yaxes(range=[0,1.35], showgrid=False, visible=False)
    fig.update_layout(height=170, margin=dict(l=10,r=10,t=10,b=10))
    return fig


# ---------------------- LIVE SIGNAL (1m) PANEL ----------------------
@st.cache_data(ttl=60)
def ls_fetch_and_signal(symbol: str, interval: str = "1m", period: str = "1d",
                       display_bars: int = 200,
                       tp_pips: float = 12, sl_pips: float = 8,
                       pullback_pips: float = 2, confirm_bars: int = 3) -> pd.DataFrame:
    df = yf.download(symbol, interval=interval, period=period, auto_adjust=False, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Ichimoku
    df['tenkan'] = (df['High'].rolling(9).max() + df['Low'].rolling(9).min()) / 2
    df['kijun']  = (df['High'].rolling(26).max() + df['Low'].rolling(26).min()) / 2
    df['spanA']  = ((df['tenkan'] + df['kijun']) / 2).shift(26)
    # EMA200
    df['EMA200'] = df['Close'].ewm(span=200).mean()
    # RSI14
    delta  = df['Close'].diff()
    gain   = delta.clip(lower=0)
    loss   = -delta.clip(upper=0)
    avg_g  = gain.rolling(14).mean()
    avg_l  = loss.rolling(14).mean()
    df['RSI14'] = 100 - (100 / (1 + avg_g / avg_l))

    # Ham sinyal
    cond_b = (
        (df['Close'] > df['spanA']) &
        (df['tenkan'] > df['kijun']) &
        (df['Close'] > df['EMA200']) &
        (df['RSI14'] > 50)
    )
    cond_s = (
        (df['Close'] < df['spanA']) &
        (df['tenkan'] < df['kijun']) &
        (df['Close'] < df['EMA200']) &
        (df['RSI14'] < 50)
    )
    df['RawSignal'] = ''
    df.loc[cond_b, 'RawSignal'] = 'BUY'
    df.loc[cond_s, 'RawSignal'] = 'SELL'

    # Bar teyidi
    df['Signal'] = ''
    for side in ('BUY','SELL'):
        ok = (df['RawSignal'] == side).rolling(confirm_bars).sum() == confirm_bars
        df.loc[ok, 'Signal'] = side
    df['Signal'] = df['Signal'].where(df['Signal'] != df['Signal'].shift(), '')

    pip_val = get_pip_value(symbol)
    df['EntryPrice'] = df['Close']
    df.loc[df['Signal']=='BUY',  'EntryPrice'] = df['Close'] - pullback_pips * pip_val
    df.loc[df['Signal']=='SELL', 'EntryPrice'] = df['Close'] + pullback_pips * pip_val

    return df.tail(display_bars)


def plot_live_panel(symbol: str, df: pd.DataFrame, tp_pips: float = 12, sl_pips: float = 8):
    fig = go.Figure()
    if df is None or df.empty:
        fig.update_layout(height=300, title="Veri yok")
        return fig

    # TZ
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC").tz_convert(TR_TZ)
    else:
        df.index = df.index.tz_convert(TR_TZ)

    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close', line=dict(width=1)))

    signals = df[df['Signal'] != '']
    pip = get_pip_value(symbol)

    # İşaretler
    buys  = signals[signals['Signal']=='BUY']
    sells = signals[signals['Signal']=='SELL']
    fig.add_trace(go.Scatter(x=buys.index,  y=df.loc[buys.index,'Close'], mode='markers', name='BUY',
                             marker=dict(symbol='triangle-up', size=10, color='green', line=dict(color='black', width=1))))
    fig.add_trace(go.Scatter(x=sells.index, y=df.loc[sells.index,'Close'], mode='markers', name='SELL',
                             marker=dict(symbol='triangle-down', size=10, color='red', line=dict(color='black', width=1))))

    # Son sinyal için SL/TP bölgesi
    if not signals.empty:
        last = signals.iloc[-1]
        price_last = float(last['Close'])
        side = last['Signal']
        if side == 'BUY':
            tp = price_last + tp_pips * pip
            sl = price_last - sl_pips * pip
            col = 'green'
        else:
            tp = price_last - tp_pips * pip
            sl = price_last + sl_pips * pip
            col = 'red'
        fig.add_hline(y=tp, line_dash='dash', line_color=col)
        fig.add_hline(y=sl, line_dash='dash', line_color=col)
        fig.add_annotation(x=df.index[-1], y=tp, text=f"TP {tp:.4f}", showarrow=False, font=dict(color=col))
        fig.add_annotation(x=df.index[-1], y=sl, text=f"SL {sl:.4f}", showarrow=False, font=dict(color=col))

    fig.update_layout(height=340, margin=dict(l=30,r=20,t=40,b=20), title=f"{symbol} – Canlı Sinyal (1m)")
    return fig


# ================================ UI ================================
left, right = st.columns([2.1, 1.0])

with right:
    st.header("Ayarlar")
    default_symbol = st.session_state.get("symbol", "USDJPY=X")
    symbol = st.selectbox("Sembol", options=SYMBOL_LIST, index=SYMBOL_LIST.index(default_symbol) if default_symbol in SYMBOL_LIST else 1)
    new_sym = st.text_input("Elle gir (Yahoo formatı, ör: EURUSD=X)", value="")
    if new_sym:
        symbol = new_sym.strip().upper()
    st.session_state["symbol"] = symbol

    tf = st.radio("Zaman Dilimi", ["4 Saat","1 Saat","15 Dakika","5 Dakika"], index=1)

    st.markdown("<span class='small-muted'>Veri TTL 60 sn. 'Yenile' ile anında güncelleyebilirsiniz.</span>", unsafe_allow_html=True)
    if st.button("Yenile"):
        fetch_ohlc.clear()
        fetch_last_price.clear()
        ls_fetch_and_signal.clear()
        st.experimental_rerun()

    # Fiyat kutusu
    st.subheader("Fiyat")
    price = fetch_last_price(symbol)
    base_key = f"base_{symbol}"
    if base_key not in st.session_state:
        st.session_state[base_key] = price
    base_price = st.session_state[base_key]
    if price is None or base_price is None or base_price == 0:
        st.info("Fiyat alınamadı.")
        pct_txt = "-"
    else:
        pct = 100 * (price - base_price) / base_price
        pct_txt = f"{'+' if pct>0 else ''}{pct:.2f}%"
    st.metric(label="Güncel Fiyat", value=f"{price:.5f}" if price else "-", delta=pct_txt)

with left:
    st.title("Forex Analyzer (Web)")

    # Ana grafik
    intv_map = {"4 Saat":"4h", "1 Saat":"60m", "15 Dakika":"15m", "5 Dakika":"5m"}
    per_map  = {"4 Saat":"30d", "1 Saat":"7d",  "15 Dakika":"5d",   "5 Dakika":"1d"}
    fig, df = plot_main_figure(symbol, intv_map[tf], per_map[tf])
    st.plotly_chart(fig, use_container_width=True)

    # Analiz tabloları + Gauge
    tech_df, dec_df = analyse_symbol(symbol)
    colA, colB = st.columns([1.3, 1])
    with colA:
        st.subheader("Teknik Özet")
        st.dataframe(tech_df, use_container_width=True, height=200)
        st.subheader("Karar Tablosu")
        st.dataframe(dec_df, use_container_width=True, height=200)
    with colB:
        karar = final_global_decision(dec_df) if not dec_df.empty else "Bekle"
        st.subheader("Gauge")
        st.plotly_chart(gauge_figure(karar), use_container_width=True)
        st.markdown(f"**Genel Karar:** {karar}")

    # Canlı tetikleyici paneli
    st.subheader("Canlı Ichimoku‑EMA200‑RSI Paneli (1m)")
    live_df = ls_fetch_and_signal(symbol)
    st.plotly_chart(plot_live_panel(symbol, live_df), use_container_width=True)

st.caption("Not: Bu çalışma eğitim amaçlıdır; yatırım tavsiyesi değildir.")
