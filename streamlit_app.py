import time
import requests
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

# --- Configuration (must be first Streamlit command) ---
st.set_page_config(page_title='LSE Swing Scanner', layout='wide')

# Finnhub API Key for OHLCV data
FINNHUB_API_KEY = 'd12acb9r01qmhi3heaqgd12acb9r01qmhi3hear0'

# --- Fetch LSE Tickers via Wikipedia ---
@st.cache_data(show_spinner=False)
def fetch_lse_tickers() -> list[str]:
    urls = {
        'FTSE 100': 'https://en.wikipedia.org/wiki/FTSE_100',
        'FTSE 250': 'https://en.wikipedia.org/wiki/FTSE_250'
    }
    tickers = []
    for name, url in urls.items():
        try:
            tables = pd.read_html(url)
            for df in tables:
                for col in df.columns:
                    if str(col).lower() in ('epic', 'ticker', 'code'):
                        tickers += df[col].astype(str).str.upper().tolist()
                        raise StopIteration
        except StopIteration:
            continue
        except Exception as e:
            st.warning(f"Could not scrape {name}: {e}")
    return sorted(set(tickers))

# --- Fetch OHLCV via Finnhub HTTP API ---
@st.cache_data(show_spinner=False)
def fetch_ohlcv(symbol: str, resolution: str = 'D', days: int = 365*5) -> pd.DataFrame:
    url = 'https://finnhub.io/api/v1/stock/candle'
    end = int(time.time())
    start = end - days * 24 * 3600
    params = {'symbol': symbol, 'resolution': resolution, 'from': start, 'to': end, 'token': FINNHUB_API_KEY}
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data.get('s') != 'ok':
            return pd.DataFrame()
        df = pd.DataFrame({
            'time': pd.to_datetime(data['t'], unit='s'),
            'open': data['o'], 'high': data['h'], 'low': data['l'], 'close': data['c'], 'volume': data['v']
        }).set_index('time')
        return df
    except Exception as e:
        st.warning(f"Error fetching data for {symbol}: {e}")
        return pd.DataFrame()

# --- Feature Engineering without pandas_ta ---
@st.cache_data(show_spinner=False)
def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # EMA
    df['ema10'] = df['close'].ewm(span=10, adjust=False).mean()
    df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
    # RSI
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['rsi14'] = 100 - (100 / (1 + rs))
    # ATR
    high_low = df['high'] - df['low']
    high_prev = (df['high'] - df['close'].shift()).abs()
    low_prev = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_prev, low_prev], axis=1).max(axis=1)
    df['atr14'] = tr.rolling(window=14).mean()
    # Momentum
    df['mom5'] = df['close'].pct_change(5)
    return df.dropna()

# --- Load Model ---
@st.cache_resource(show_spinner=False)
def load_model():
    return CalibratedClassifierCV(RandomForestClassifier(n_estimators=50), cv=3)

# --- Streamlit App ---
st.title('LSE Swing Trade Scanner')

# Sidebar
threshold = st.sidebar.slider('Strength threshold (%)', 0.0, 100.0, 70.0)
max_tickers = st.sidebar.number_input('Max tickers to display', 1, 100, 20)

if st.sidebar.button('Initialize Tickers'):
    tickers = fetch_lse_tickers()
    st.session_state['tickers'] = tickers
    st.success(f"Loaded {len(tickers)} LSE tickers from Wikipedia")

if 'tickers' not in st.session_state:
    st.info("Click 'Initialize Tickers' to load tickers.")
    st.stop()

# --- Scanner ---
def scan_tickers() -> pd.DataFrame:
    model = load_model()
    results = []
    for symbol in st.session_state['tickers'][:max_tickers]:
        df = fetch_ohlcv(symbol)
        if df.empty or len(df) < 60:
            continue
        feats = compute_features(df)
        X = feats[['ema10', 'ema50', 'rsi14', 'atr14', 'mom5']]
        y = (feats['close'].shift(-10) >= feats['close'] * 1.05).astype(int)
        model.fit(X[:-30], y[:-30])
        prob = model.predict_proba(X.iloc[[-1]])[:,1][0] * 100
        if prob >= threshold:
            entry = feats['close'].iloc[-1]
            stop = entry - 1.5 * feats['atr14'].iloc[-1]
            results.append({
                'Symbol': symbol,
                'Strength (%)': round(prob,1),
                'Entry Price': round(entry,3),
                'Stop-Loss': round(stop,3)
            })
    return pd.DataFrame(results)

if st.button('Run Scanner'):
    df_out = scan_tickers()
    if df_out.empty:
        st.info('No tickers met the threshold.')
    else:
        st.dataframe(df_out.sort_values('Strength (%)', ascending=False))
