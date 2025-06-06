
import os
import time
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
import finnhub
import pandas_ta as ta

# Config
FINNHUB_API_KEY = os.getenv('FINNHUB_API_KEY')
client = finnhub.Client(api_key=FINNHUB_API_KEY)

def fetch_lse_tickers():
    symbols = client.stock_symbols('GB')
    return [s['symbol'] for s in symbols if s.get('exchange') == 'XLON']

def fetch_ohlcv(symbol, resolution='D', days=365*5):
    end = int(time.time())
    start = end - days * 24 * 3600
    r = client.stock_candles(symbol, resolution, start, end)
    if r.get('s') != 'ok':
        return None
    df = pd.DataFrame({
        'time': pd.to_datetime(r['t'], unit='s'),
        'open': r['o'],
        'high': r['h'],
        'low': r['l'],
        'close': r['c'],
        'volume': r['v']
    }).set_index('time')
    return df

def compute_features(df):
    df['ema10'] = ta.ema(df['close'], length=10)
    df['ema50'] = ta.ema(df['close'], length=50)
    df['rsi14'] = ta.rsi(df['close'], length=14)
    df['atr14'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    df['mom5'] = df['close'].pct_change(5)
    return df.dropna()

def compute_strength(X, model):
    return (model.predict_proba(X)[:, 1] * 100).round(1)

def compute_entry_stop(df, idx):
    e = df['close'].iloc[idx]
    return e, e - 1.5 * df['atr14'].iloc[idx]

@st.cache(allow_output_mutation=True)
def load_model():
    return CalibratedClassifierCV(RandomForestClassifier(n_estimators=50), cv=3)

st.set_page_config(page_title='LSE Swing Scanner')
st.title('LSE Swing Scanner')
th = st.sidebar.slider('Strength (%)', 0.0, 100.0, 70.0)
mt = st.sidebar.number_input('Max tickers', 1, 100, 20)

if st.sidebar.button('Init'):
    st.session_state['t'] = fetch_lse_tickers()
    st.success(f"Loaded {len(st.session_state['t'])} tickers.")

if 't' not in st.session_state:
    st.info('Please click Init to load tickers.')
    st.stop()

def scan():
    m = load_model()
    res = []
    for s in st.session_state['t'][:mt]:
        df = fetch_ohlcv(s)
        if df is None or len(df) < 60:
            continue
        df = compute_features(df)
        X = df[['ema10', 'ema50', 'rsi14', 'atr14', 'mom5']]
        y = (df['close'].shift(-10) >= df['close'] * 1.05).astype(int)
        m.fit(X[:-30], y[:-30])
        strg = compute_strength(X.iloc[[-1]], m)[0]
        if strg >= th:
            e, stop = compute_entry_stop(df, -1)
            res.append({'Sym': s, 'Str': strg, 'Ent': round(e, 3), 'Stop': round(stop, 3)})
    return pd.DataFrame(res)

if st.button('Run'):
    df = scan()
    if df.empty:
        st.info('No signals found.')
    else:
        st.dataframe(df)
