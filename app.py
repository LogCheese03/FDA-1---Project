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
end_date = st.sidebar.date_input(
    "End Date",
    value=date.today(),
    min_value=start_date + timedelta(days=365),
)

if len(tickers) < 2:
    st.info("Select 2 to 5 stock tickers to compare.")
    st.stop()

if (end_date - start_date).days < 365:
    st.sidebar.error("Date range must be at least 1 year.")
    st.stop()

# -- Data download --------------------------------------------
@st.cache_data(show_spinner="Fetching data…", ttl=3600)
def load_data(tickers: tuple[str, ...], start: date, end: date) -> pd.DataFrame:
    """
    Download adjusted close prices for all tickers + S&P 500 benchmark.
    Returns a DataFrame of Close columns, one per symbol.
    """
    symbols = list(tickers) + ["^GSPC"]
    raw = yf.download(symbols, start=start, end=end, progress=False, auto_adjust=True)

    # Extract just the Close (auto_adjust=True makes Close == Adj Close)
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"].copy()
    else:
        close = raw[["Close"]].copy()
        close.columns = [tickers[0]]

    return close


@st.cache_data(ttl=3600)
def validate_and_align(
    close: pd.DataFrame,
    tickers: tuple[str, ...],
    missing_threshold: float = 0.05,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    """
    Validates per-ticker data quality and aligns to the overlapping date range.

    Returns:
        ticker_close  – aligned close prices for valid user tickers
        bench_close   – aligned close prices for ^GSPC
        dropped       – tickers removed due to >5% missing data
        warnings      – non-fatal messages to surface to the user
    """
    warnings_out: list[str] = []
    dropped: list[str] = []

    # ── 1. Separate benchmark from user tickers ──────────────────────────────
    user_cols = [t for t in tickers if t in close.columns]
    missing_tickers = [t for t in tickers if t not in close.columns]

    if missing_tickers:
        dropped.extend(missing_tickers)
        warnings_out.append(
            f"Could not download data for: **{', '.join(missing_tickers)}**. "
            "They have been removed from the analysis."
        )

    bench = close["^GSPC"].to_frame() if "^GSPC" in close.columns else pd.DataFrame()

    if bench.empty:
        warnings_out.append("Could not download S&P 500 benchmark data (^GSPC).")

    user_close = close[user_cols].copy()

    # ── 2. Drop tickers with > threshold missing values ──────────────────────
    for tkr in user_cols:
        missing_pct = user_close[tkr].isna().mean()
        if missing_pct > missing_threshold:
            dropped.append(tkr)
            warnings_out.append(
                f"**{tkr}** has {missing_pct:.1%} missing values "
                f"(threshold: {missing_threshold:.0%}) and has been removed."
            )

    user_close = user_close.drop(columns=dropped, errors="ignore")

    if user_close.empty:
        return pd.DataFrame(), bench, dropped, warnings_out

    # ── 3. Truncate to overlapping date range ────────────────────────────────
    if not bench.empty:
        combined = user_close.join(bench, how="outer")
    else:
        combined = user_close

    first_valid = max(
        combined[col].first_valid_index()
        for col in combined.columns
        if combined[col].first_valid_index() is not None
    )
    last_valid = min(
        combined[col].last_valid_index()
        for col in combined.columns
        if combined[col].last_valid_index() is not None
    )

    original_start = user_close.index.min()
    original_end   = user_close.index.max()

    user_close = user_close.loc[first_valid:last_valid].copy()
    if not bench.empty:
        bench = bench.loc[first_valid:last_valid].copy()

    if first_valid > original_start or last_valid < original_end:
        warnings_out.append(
            f"Data truncated to overlapping range: "
            f"**{first_valid.date()}** → **{last_valid.date()}**."
        )

    # ── 4. Forward-fill any remaining gaps (e.g. staggered holidays) ─────────
    user_close = user_close.ffill()
    if not bench.empty:
        bench = bench.ffill()

    return user_close, bench, dropped, warnings_out


# -- Run download + validation --------------------------------
try:
    raw_close = load_data(tuple(tickers), start_date, end_date)
except Exception as e:
    st.error(f"Download failed: {e}")
    st.stop()

if raw_close.empty:
    st.error("No data returned. Check your ticker symbols and date range.")
    st.stop()

user_close, bench_close, dropped_tickers, data_warnings = validate_and_align(
    raw_close, tuple(tickers)
)

# Surface any warnings to the user
for w in data_warnings:
    st.warning(w)

# If too many tickers were dropped, we can't continue
remaining = [t for t in tickers if t not in dropped_tickers]
if len(remaining) < 2:
    st.error(
        "Fewer than 2 tickers have sufficient data for the selected date range. "
        "Please adjust your selection or date range."
    )
    st.stop()

# -- Build price_data dict ------------------------------------
# Includes user tickers first, then S&P 500 benchmark last
price_data: dict[str, pd.DataFrame] = {}

for tkr in remaining:
    df = user_close[[tkr]].copy()
    df.columns = ["Close"]
    df["Daily Return"] = df["Close"].pct_change()
    price_data[tkr] = df

# Add S&P 500 benchmark
if not bench_close.empty:
    df = bench_close.copy()
    df.columns = ["Close"]
    df["Daily Return"] = df["Close"].pct_change()
    price_data["^GSPC"] = df

if not price_data:
    st.error("No usable data was downloaded for the selected tickers.")
    st.stop()

# -- Key Metrics (one column group per ticker + benchmark) ----
st.subheader("Key Metrics")

cols = st.columns(len(price_data))
for col, (tkr, df) in zip(cols, price_data.items()):
    latest_close  = float(df["Close"].iloc[-1])
    total_return  = float((df["Close"].iloc[-1] / df["Close"].iloc[0]) - 1)
    ann_vol       = float(df["Daily Return"].std() * math.sqrt(252))
    period_high   = float(df["Close"].max())
    period_low    = float(df["Close"].min())

    col.markdown(f"**{tkr}**")
    # S&P 500 is in index points, not USD — label accordingly
    unit = "pts" if tkr == "^GSPC" else "USD"
    col.metric("Latest Close",          f"{latest_close:,.2f} {unit}")
    col.metric("Period Return",         f"{total_return:.2%}")
    col.metric("Annualised Volatility", f"{ann_vol:.2%}")
    col.metric("Period High",           f"{period_high:,.2f} {unit}")
    col.metric("Period Low",            f"{period_low:,.2f} {unit}")

st.divider()

# -- Closing Price Chart --------------------------------------
st.subheader("Closing Price")
st.caption("S&P 500 (^GSPC) is shown in index points; all other tickers in USD.")

fig_price = go.Figure()
for tkr, df in price_data.items():
    fig_price.add_trace(
        go.Scatter(
            x=df.index, y=df["Close"], mode="lines", name=tkr,
            line=dict(width=2 if tkr == "^GSPC" else 1.5),
        )
    )
fig_price.update_layout(
    yaxis_title="Price (USD) / Index Points",
    xaxis_title="Date",
    template="plotly_white",
    height=450,
    legend_title="Ticker",
)
st.plotly_chart(fig_price, use_container_width=True)

st.divider()

# -- Normalised Returns Chart (base = 100 on first day) -------
st.subheader("Normalised Returns (base = 100)")
st.caption("All series rebased to 100 on the first trading day — directly comparable regardless of price scale.")

fig_norm = go.Figure()
for tkr, df in price_data.items():
    normalised = df["Close"] / df["Close"].iloc[0] * 100
    fig_norm.add_trace(
        go.Scatter(
            x=df.index, y=normalised, mode="lines", name=tkr,
            line=dict(width=2 if tkr == "^GSPC" else 1.5),
        )
    )
fig_norm.update_layout(
    yaxis_title="Indexed Price (base = 100)",
    xaxis_title="Date",
    template="plotly_white",
    height=400,
    legend_title="Ticker",
)
st.plotly_chart(fig_norm, use_container_width=True)