import os
import time
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
import finnhub
import pandas_ta as ta

# --- Configuration (must be first Streamlit command) ---
st.set_page_config(page_title='LSE Swing Scanner', layout='wide')

# Initialize Finnhub client
FINNHUB_API_KEY = 'd0rnqspr01qumepfd80gd0rnqspr01qumepfd810'
client = finnhub.Client(api_key=FINNHUB_API_KEY)

# --- Caching functions ---
@st.cache_data(show_spinner=False)
def fetch_lse_tickers():
    try:
        symbols = client.stock_symbols('GB')
        return [s['symbol'] for s in symbols if s.get('exchange') == 'XLON']
    except Exception as e:
        st.error(f"Error fetching LSE tickers: {e}")
        return []

# Next function
def fetch_ohlcv(symbol: str, resolution: str = 'D', days: int = 365*5) -> pd.DataFrame:
    end = int(time.time())
    start = end - days * 24 * 3600
    r = client.stock_candles(symbol, resolution, start, end)
    if r.get('s') != 'ok':
        return pd.DataFrame()
    df = pd.DataFrame({
        'time': pd.to_datetime(r['t'], unit='s'),
        'open': r['o'],
        'high': r['h'],
        'low': r['l'],
        'close': r['c'],
        'volume': r['v']
    }).set_index('time')
    return df

@st.cache_data(show_spinner=False)
def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['ema10'] = ta.ema(df['close'], length=10)
    df['ema50'] = ta.ema(df['close'], length=50)
    df['rsi14'] = ta.rsi(df['close'], length=14)
    df['atr14'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    df['mom5'] = df['close'].pct_change(5)
    return df.dropna()

@st.cache_resource(show_spinner=False)
def load_model():
    return CalibratedClassifierCV(RandomForestClassifier(n_estimators=50), cv=3)

# --- Streamlit UI ---
st.title('LSE Swing Trade Scanner')

# Sidebar controls
threshold = st.sidebar.slider('Strength threshold (%)', 0.0, 100.0, 70.0)
max_tickers = st.sidebar.number_input('Max tickers to display', 1, 100, 20)

if st.sidebar.button('Initialize Tickers'):
    tickers = fetch_lse_tickers()
    st.session_state['tickers'] = tickers
    st.success(f"Loaded {len(tickers)} tickers.")

if 'tickers' not in st.session_state:
    st.info("Click 'Initialize Tickers' to load tickers.")
    st.stop()

# Main scan function
def scan_tickers():
    model = load_model()
    results = []
    tickers = st.session_state['tickers']
    for symbol in tickers[:max_tickers]:
        df = fetch_ohlcv(symbol)
        if df.empty or len(df) < 60:
            continue
        df = compute_features(df)
        X = df[['ema10', 'ema50', 'rsi14', 'atr14', 'mom5']]
        y = (df['close'].shift(-10) >= df['close'] * 1.05).astype(int)
        model.fit(X[:-30], y[:-30])
        strg = model.predict_proba(X.iloc[[-1]])[:,1][0] * 100
        if strg >= threshold:
            entry = df['close'].iloc[-1]
            stop = entry - 1.5 * df['atr14'].iloc[-1]
            results.append({
                'Symbol': symbol,
                'Strength (%)': round(strg,1),
                'Entry Price': round(entry,3),
                'Stop-Loss': round(stop,3)
            })
    return pd.DataFrame(results)

if st.button('Run Scanner'):
    output_df = scan_tickers()
    if output_df.empty:
        st.info('No tickers met the threshold.')
    else:
        st.dataframe(output_df.sort_values('Strength (%)', ascending=False))
