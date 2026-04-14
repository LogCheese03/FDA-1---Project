# app.py
# -------------------------------------------------------
# A Streamlit stock analysis dashboard supporting 2-5 tickers.
# Run with:  uv run streamlit run app.py
# -------------------------------------------------------

import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import date, timedelta
import math

st.set_page_config(page_title="Stock Analyzer", layout="wide")
st.title("Stock Analysis Dashboard")

# -- Sidebar --------------------------------------------------
st.sidebar.header("Settings")
ticker_options = ["AAPL", "MSFT", "NVDA", "TSLA", "DKNG"]

tickers = st.sidebar.multiselect(
    "Select 2 – 5 stock tickers",
    options=ticker_options,
    max_selections=5,
)

default_start = date.today() - timedelta(days=365)
start_date = st.sidebar.date_input("Start Date", value=default_start, min_value=date(1970, 1, 1))
end_date   = st.sidebar.date_input("End Date",   value=date.today(), min_value=date(1970, 1, 1))

if len(tickers) < 2:
    st.info("Select 2 to 5 stock tickers to compare.")
    st.stop()

if start_date >= end_date:
    st.sidebar.error("Start date must be before end date.")
    st.stop()

# -- Data download --------------------------------------------
@st.cache_data(show_spinner="Fetching data…", ttl=3600)
def load_data(tickers: tuple[str, ...], start: date, end: date) -> pd.DataFrame:
    """Download daily OHLCV data; returns a tidy DataFrame with a (field, ticker) MultiIndex."""
    return yf.download(list(tickers), start=start, end=end, progress=False)

try:
    raw = load_data(tuple(tickers), start_date, end_date)
except Exception as e:
    st.error(f"Failed to download data: {e}")
    st.stop()

if raw.empty:
    st.error("No data returned. Check your ticker symbols and date range.")
    st.stop()

# -- Build a clean per-ticker dictionary ----------------------
# yfinance returns a MultiIndex (field, ticker) for multiple symbols.
# We split it into {ticker: DataFrame} so every chart/metric loop is simple.
price_data: dict[str, pd.DataFrame] = {}

if isinstance(raw.columns, pd.MultiIndex):
    for tkr in tickers:
        try:
            df = raw.xs(tkr, axis=1, level=1).copy()
        except KeyError:
            st.warning(f"No data found for **{tkr}** – skipping.")
            continue
        df["Daily Return"] = df["Close"].pct_change()
        price_data[tkr] = df
else:
    # Single-ticker fallback (yfinance drops the MultiIndex when len==1,
    # but the sidebar already blocks <2, so this path shouldn't be hit in
    # production. Kept for safety.)
    df = raw.copy()
    df["Daily Return"] = df["Close"].pct_change()
    price_data[tickers[0]] = df

if not price_data:
    st.error("No usable data was downloaded for the selected tickers.")
    st.stop()

# -- Key Metrics (one column group per ticker) ----------------
st.subheader("Key Metrics")

cols = st.columns(len(price_data))
for col, (tkr, df) in zip(cols, price_data.items()):
    latest_close  = float(df["Close"].iloc[-1])
    total_return  = float((df["Close"].iloc[-1] / df["Close"].iloc[0]) - 1)
    ann_vol       = float(df["Daily Return"].std() * math.sqrt(252))
    period_high   = float(df["Close"].max())
    period_low    = float(df["Close"].min())

    col.markdown(f"**{tkr}**")
    col.metric("Latest Close",           f"${latest_close:,.2f}")
    col.metric("Period Return",          f"{total_return:.2%}")
    col.metric("Annualised Volatility",  f"{ann_vol:.2%}")
    col.metric("Period High",            f"${period_high:,.2f}")
    col.metric("Period Low",             f"${period_low:,.2f}")

st.divider()

# -- Closing Price Chart (all tickers on one chart) -----------
st.subheader("Closing Price")

fig_price = go.Figure()
for tkr, df in price_data.items():
    fig_price.add_trace(
        go.Scatter(x=df.index, y=df["Close"], mode="lines", name=tkr, line=dict(width=1.5))
    )
fig_price.update_layout(
    yaxis_title="Price (USD)", xaxis_title="Date",
    template="plotly_white", height=450, legend_title="Ticker",
)
st.plotly_chart(fig_price, use_container_width=True)

st.divider()

# -- Normalised Returns Chart (base = 100 on first day) -------
st.subheader("Normalised Returns (base = 100)")

fig_norm = go.Figure()
for tkr, df in price_data.items():
    normalised = df["Close"] / df["Close"].iloc[0] * 100
    fig_norm.add_trace(
        go.Scatter(x=df.index, y=normalised, mode="lines", name=tkr, line=dict(width=1.5))
    )
fig_norm.update_layout(
    yaxis_title="Indexed Price", xaxis_title="Date",
    template="plotly_white", height=400, legend_title="Ticker",
)
st.plotly_chart(fig_norm, use_container_width=True)