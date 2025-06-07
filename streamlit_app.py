import os
import time
import requests
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
import pandas_ta as ta

# --- Configuration (must be first Streamlit command) ---
st.set_page_config(page_title='LSE Swing Scanner', layout='wide')

# Finnhub API Key
FINNHUB_API_KEY = 'd12acb9r01qmhi3heaqgd12acb9r01qmhi3hear0'

# --- Data Fetching via HTTP ---
@st.cache_data(show_spinner=False)
def fetch_lse_tickers() -> list[str]:
    """
    Retrieve LSE tickers via Finnhub HTTP API.
    """
    url = "https://finnhub.io/api/v1/stock/symbol"
    params = {"exchange": "XLON", "token": FINNHUB_API_KEY}
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        symbols = resp.json()
        return [s['symbol'] for s in symbols if s.get('exchange') == 'XLON']
    except Exception as e:
        st.error(f"Error fetching LSE tickers: {e}")
        return []

@st.cache_data(show_spinner=False)
def fetch_ohlcv(symbol: str, resolution: str = 'D', days: int = 365*5) -> pd.DataFrame:
    """
    Fetch OHLCV data via Finnhub HTTP API.
    """
    url = "https://finnhub.io/api/v1/stock/candle"
    end = int(time.time())
    start = end - days * 24 * 3600
    params = {
        'symbol': symbol,
        'resolution': resolution,
        'from': start,
        'to': end,
        'token': FINNHUB_API_KEY
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data.get('s') != 'ok':
            return pd.DataFrame()
        df = pd.DataFrame({
            'time': pd.to_datetime(data['t'], unit='s'),
            'open': data['o'],
            'high': data['h'],
            'low': data['l'],
            'close': data['c'],
            'volume': data['v']
        }).set_index('time')
        return df
    except Exception as e:
        st.warning(f"Error fetching data for {symbol}: {e}")
        return pd.DataFrame()

# --- Feature Engineering ---
@st.cache_data(show_spinner=False)
def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['ema10'] = ta.ema(df['close'], length=10)
    df['ema50'] = ta.ema(df['close'], length=50)
    df['rsi14'] = ta.rsi(df['close'], length=14)
    df['atr14'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    df['mom5'] = df['close'].pct_change(5)
    return df.dropna()

# --- Model Loading ---
@st.cache_resource(show_spinner=False)
def load_model():
    return CalibratedClassifierCV(RandomForestClassifier(n_estimators=50), cv=3)

# --- Streamlit App UI ---
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

# --- Core Scan Function ---
def scan_tickers() -> pd.DataFrame:
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
        prob = model.predict_proba(X.iloc[[-1]])[:,1][0] * 100
        if prob >= threshold:
            entry = df['close'].iloc[-1]
            stop = entry - 1.5 * df['atr14'].iloc[-1]
            results.append({
                'Symbol': symbol,
                'Strength (%)': round(prob, 1),
                'Entry Price': round(entry, 3),
                'Stop-Loss': round(stop, 3)
            })
    return pd.DataFrame(results)

if st.button('Run Scanner'):
    df_out = scan_tickers()
    if df_out.empty:
        st.info('No tickers met the threshold.')
    else:
        st.dataframe(df_out.sort_values('Strength (%)', ascending=False))
