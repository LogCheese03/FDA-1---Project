# app.py
# -------------------------------------------------------
# Stock Comparison and Analysis Dashboard
# Run with:  uv run streamlit run app.py
# -------------------------------------------------------

import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from datetime import date, timedelta
from scipy import stats as scipy_stats

st.set_page_config(page_title="Stock Analysis Dashboard", layout="wide")
st.title("Stock Analysis Dashboard")

# ── Sidebar ──────────────────────────────────────────────────────
st.sidebar.header("Settings")

raw_input = st.sidebar.text_input(
    "Enter 2–5 stock tickers (comma-separated)",
    value="AAPL, MSFT, NVDA",
    help="Any valid yfinance ticker, e.g. AAPL, BTC-USD, MSFT",
)
tickers = [t.strip().upper() for t in raw_input.split(",") if t.strip()]

default_start = date.today() - timedelta(days=365 * 3)
start_date = st.sidebar.date_input("Start Date", value=default_start, min_value=date(1970, 1, 1))
end_date = st.sidebar.date_input("End Date", value=date.today())

# ── Input validation ─────────────────────────────────────────────
if len(tickers) < 2:
    st.info("Enter 2 to 5 ticker symbols to get started.")
    st.stop()

if len(tickers) > 5:
    st.error(f"You entered {len(tickers)} tickers. Please enter no more than 5.")
    st.stop()

if (end_date - start_date).days < 365:
    st.sidebar.error("Enter a valid date outside of the one year minimum.")
    st.stop()

# ── Data functions ────────────────────────────────────────────────
@st.cache_data(show_spinner="Fetching data…", ttl=3600)
def load_data(tickers: tuple[str, ...], start: date, end: date) -> pd.DataFrame:
    symbols = list(tickers) + ["^GSPC"]
    raw = yf.download(symbols, start=start, end=end, progress=False, auto_adjust=True)
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"].copy()
    else:
        close = raw[["Close"]].copy()
        close.columns = [tickers[0]]

    # yfinance batch downloads occasionally return all-NaN for a valid ticker.
    # Retry those individually so they don't get falsely flagged as bad data.
    for sym in symbols:
        if sym in close.columns and close[sym].isna().all():
            try:
                single = yf.download(sym, start=start, end=end, progress=False, auto_adjust=True)
                if not single.empty:
                    close[sym] = single["Close"]
            except Exception:
                pass

    return close

@st.cache_data(ttl=3600)
def validate_and_align(
    close: pd.DataFrame,
    tickers: tuple[str, ...],
    missing_threshold: float = 0.05,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    warnings_out: list[str] = []
    dropped: list[str] = []

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
    original_end = user_close.index.max()

    user_close = user_close.loc[first_valid:last_valid].copy()
    if not bench.empty:
        bench = bench.loc[first_valid:last_valid].copy()

    if first_valid > original_start or last_valid < original_end:
        warnings_out.append(
            f"Data truncated to overlapping range: "
            f"**{first_valid.date()}** → **{last_valid.date()}**."
        )

    user_close = user_close.ffill()
    if not bench.empty:
        bench = bench.ffill()

    return user_close, bench, dropped, warnings_out


# ── Load & validate ───────────────────────────────────────────────
try:
    with st.spinner("Downloading price data…"):
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

for w in data_warnings:
    st.warning(w)

remaining = [t for t in tickers if t not in dropped_tickers]
if len(remaining) < 2:
    st.error(
        "Fewer than 2 tickers have sufficient data for the selected date range. "
        "Please adjust your selection or date range."
    )
    st.stop()

# ── Build price_data ──────────────────────────────────────────────
price_data: dict[str, pd.DataFrame] = {}

for tkr in remaining:
    df = user_close[[tkr]].copy()
    df.columns = ["Close"]
    df["Daily Return"] = df["Close"].pct_change()
    price_data[tkr] = df

if not bench_close.empty:
    df = bench_close.copy()
    df.columns = ["Close"]
    df["Daily Return"] = df["Close"].pct_change()
    price_data["^GSPC"] = df

if not price_data:
    st.error("No usable data was downloaded for the selected tickers.")
    st.stop()

# ── Sidebar: stock selector (drives tabs 2-4) ─────────────────────
user_tickers = [t for t in price_data if t != "^GSPC"]

selected_stocks = st.sidebar.multiselect(
    "Stocks to include in analysis",
    options=user_tickers,
    default=user_tickers,
    help="Toggle stocks on/off in the analysis tabs.",
)

if not selected_stocks:
    st.warning("Select at least one stock in the sidebar to display the analysis.")
    st.stop()

show_tickers = selected_stocks + (["^GSPC"] if "^GSPC" in price_data else [])

INITIAL_INVESTMENT = 10_000

# ── Tabs ──────────────────────────────────────────────────────────
tab_overview, tab_returns, tab_risk, tab_corr, tab_about = st.tabs([
    "Overview",
    "Price & Returns",
    "Risk & Distribution",
    "Correlation & Portfolio",
    "About",
])

# ════════════════════════════════════════════════════════════════
# TAB 1 – Overview
# ════════════════════════════════════════════════════════════════
with tab_overview:
    st.header("Key Metrics")

    cols = st.columns(len(price_data))
    for col, (tkr, df) in zip(cols, price_data.items()):
        latest_close = float(df["Close"].iloc[-1])
        total_return = float((df["Close"].iloc[-1] / df["Close"].iloc[0]) - 1)
        ann_vol = float(df["Daily Return"].std() * math.sqrt(252))
        period_high = float(df["Close"].max())
        period_low = float(df["Close"].min())

        col.markdown(f"**{tkr}**")
        unit = "pts" if tkr == "^GSPC" else "USD"
        col.metric("Latest Close", f"{latest_close:,.2f} {unit}")
        col.metric("Period Return", f"{total_return:.2%}")
        col.metric("Annualised Volatility", f"{ann_vol:.2%}")
        col.metric("Period High", f"{period_high:,.2f} {unit}")
        col.metric("Period Low", f"{period_low:,.2f} {unit}")

    st.divider()

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
        title="Adjusted Closing Prices",
        yaxis_title="Price (USD) / Index Points",
        xaxis_title="Date",
        template="plotly_white",
        height=450,
        legend_title="Ticker",
    )
    st.plotly_chart(fig_price, use_container_width=True, key="fig_price")

    st.divider()

    st.subheader("Normalised Returns (base = 100)")
    st.caption("All series rebased to 100 on the first trading day.")

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
        title="Normalised Returns (base = 100)",
        yaxis_title="Indexed Price (base = 100)",
        xaxis_title="Date",
        template="plotly_white",
        height=400,
        legend_title="Ticker",
    )
    st.plotly_chart(fig_norm, use_container_width=True, key="fig_norm")


# ════════════════════════════════════════════════════════════════
# TAB 2 – Price & Returns
# ════════════════════════════════════════════════════════════════
with tab_returns:
    st.header("Price and Return Analysis")

    # 1. Adjusted Closing Price Chart
    st.subheader("Adjusted Closing Prices")
    st.caption("S&P 500 (^GSPC) is always included as the benchmark.")

    fig_adj = go.Figure()
    for tkr in show_tickers:
        df = price_data[tkr]
        fig_adj.add_trace(
            go.Scatter(
                x=df.index, y=df["Close"], mode="lines", name=tkr,
                line=dict(width=2 if tkr == "^GSPC" else 1.5),
            )
        )
    fig_adj.update_layout(
        title="Adjusted Closing Prices",
        yaxis_title="Price (USD) / Index Points",
        xaxis_title="Date",
        template="plotly_white",
        height=450,
        legend_title="Ticker",
    )
    st.plotly_chart(fig_adj, use_container_width=True, key="fig_adj")

    st.divider()

    # 2 & 3. Summary Statistics
    st.subheader("Summary Statistics")
    st.caption(
        "Annualised figures use 252 trading days. "
        "Kurtosis is excess kurtosis (normal distribution = 0)."
    )

    rows_ret = []
    for tkr in show_tickers:
        r = price_data[tkr]["Daily Return"].dropna()
        rows_ret.append(
            {
                "Ticker": tkr,
                "Ann. Mean Return": r.mean() * 252,
                "Ann. Volatility": r.std() * math.sqrt(252),
                "Skewness": float(r.skew()),
                "Kurtosis": float(r.kurt()),
                "Min Daily Return": float(r.min()),
                "Max Daily Return": float(r.max()),
            }
        )

    stats_df = pd.DataFrame(rows_ret).set_index("Ticker")
    st.dataframe(
        stats_df.style.format(
            {
                "Ann. Mean Return": "{:.2%}",
                "Ann. Volatility": "{:.2%}",
                "Skewness": "{:.3f}",
                "Kurtosis": "{:.3f}",
                "Min Daily Return": "{:.2%}",
                "Max Daily Return": "{:.2%}",
            }
        ),
        use_container_width=True,
    )

    st.divider()

    # 4. Cumulative Wealth Index
    st.subheader("Cumulative Wealth Index ($10,000 initial investment)")
    st.caption(
        "Equal-weight portfolio return each day = simple average of all selected stocks' daily returns."
    )

    fig_wealth = go.Figure()
    for tkr in show_tickers:
        r = price_data[tkr]["Daily Return"].dropna()
        wealth = INITIAL_INVESTMENT * (1 + r).cumprod()
        fig_wealth.add_trace(
            go.Scatter(
                x=wealth.index, y=wealth, mode="lines", name=tkr,
                line=dict(width=2 if tkr == "^GSPC" else 1.5),
            )
        )

    if len(selected_stocks) >= 2:
        ew_daily = (
            pd.concat(
                [price_data[t]["Daily Return"].rename(t) for t in selected_stocks], axis=1
            )
            .dropna()
            .mean(axis=1)
        )
        ew_wealth = INITIAL_INVESTMENT * (1 + ew_daily).cumprod()
        fig_wealth.add_trace(
            go.Scatter(
                x=ew_wealth.index, y=ew_wealth, mode="lines",
                name="Equal-Weight Portfolio",
                line=dict(width=2.5, dash="dash"),
            )
        )

    fig_wealth.update_layout(
        title="Cumulative Wealth Index ($10,000 Initial Investment)",
        yaxis_title="Portfolio Value (USD)",
        xaxis_title="Date",
        template="plotly_white",
        height=450,
        legend_title="",
    )
    st.plotly_chart(fig_wealth, use_container_width=True, key="fig_wealth")


# ════════════════════════════════════════════════════════════════
# TAB 3 – Risk & Distribution
# ════════════════════════════════════════════════════════════════
with tab_risk:
    st.header("Risk and Distribution Analysis")

    # 1. Rolling Volatility
    st.subheader("Rolling Annualised Volatility")

    roll_window = st.select_slider(
        "Rolling window (trading days)",
        options=[30, 60, 90],
        value=30,
        key="roll_window",
    )

    fig_roll = go.Figure()
    for tkr in show_tickers:
        r = price_data[tkr]["Daily Return"].dropna()
        roll_vol = r.rolling(roll_window).std() * math.sqrt(252)
        fig_roll.add_trace(
            go.Scatter(
                x=roll_vol.index, y=roll_vol, mode="lines", name=tkr,
                line=dict(width=2 if tkr == "^GSPC" else 1.5),
            )
        )
    fig_roll.update_layout(
        title=f"Rolling {roll_window}-Day Annualised Volatility",
        yaxis_title="Annualised Volatility",
        xaxis_title="Date",
        yaxis_tickformat=".0%",
        template="plotly_white",
        height=400,
        legend_title="Ticker",
    )
    st.plotly_chart(fig_roll, use_container_width=True, key="fig_roll")

    st.divider()

    # 2, 3 & 4. Distribution / Q-Q / Normality test
    st.subheader("Return Distribution")

    dist_ticker = st.selectbox(
        "Select stock for distribution analysis",
        options=show_tickers,
        key="dist_ticker",
    )

    dist_returns = price_data[dist_ticker]["Daily Return"].dropna()

    jb_stat, jb_p = scipy_stats.jarque_bera(dist_returns)
    reject = jb_p < 0.05
    normality_msg = (
        f"**Jarque-Bera:** statistic = {jb_stat:.4f}, p-value = {jb_p:.4f} — "
        + ("**Rejects normality (p < 0.05)**" if reject else "**Fails to reject normality (p ≥ 0.05)**")
    )
    st.caption(normality_msg)

    hist_tab, qq_tab = st.tabs(["Histogram", "Q-Q Plot"])

    with hist_tab:
        mu, sigma = scipy_stats.norm.fit(dist_returns)
        x_range = np.linspace(dist_returns.min(), dist_returns.max(), 200)
        pdf_curve = scipy_stats.norm.pdf(x_range, mu, sigma)

        fig_hist = go.Figure()
        fig_hist.add_trace(
            go.Histogram(
                x=dist_returns,
                histnorm="probability density",
                name="Daily Returns",
                marker_color="steelblue",
                opacity=0.6,
                nbinsx=60,
            )
        )
        fig_hist.add_trace(
            go.Scatter(
                x=x_range, y=pdf_curve, mode="lines",
                name=f"Normal fit (μ={mu:.4f}, σ={sigma:.4f})",
                line=dict(color="crimson", width=2),
            )
        )
        fig_hist.update_layout(
            title=f"{dist_ticker} — Daily Return Histogram with Normal Fit",
            xaxis_title="Daily Return",
            xaxis_tickformat=".1%",
            yaxis_title="Density",
            template="plotly_white",
            height=400,
            barmode="overlay",
        )
        st.plotly_chart(fig_hist, use_container_width=True, key="fig_hist")

    with qq_tab:
        qq_res = scipy_stats.probplot(dist_returns)
        theoretical_q, ordered_vals = qq_res[0]
        slope, intercept, _ = qq_res[1]
        ref_line = slope * np.array([theoretical_q[0], theoretical_q[-1]]) + intercept

        fig_qq = go.Figure()
        fig_qq.add_trace(
            go.Scatter(
                x=theoretical_q, y=ordered_vals, mode="markers",
                name="Sample quantiles",
                marker=dict(color="steelblue", size=4, opacity=0.7),
            )
        )
        fig_qq.add_trace(
            go.Scatter(
                x=[theoretical_q[0], theoretical_q[-1]], y=ref_line, mode="lines",
                name="Normal reference",
                line=dict(color="crimson", width=2),
            )
        )
        fig_qq.update_layout(
            title=f"{dist_ticker} — Q-Q Plot vs. Normal Distribution",
            xaxis_title="Theoretical Quantiles",
            yaxis_title="Sample Quantiles",
            template="plotly_white",
            height=400,
        )
        st.plotly_chart(fig_qq, use_container_width=True, key="fig_qq")

    st.divider()

    # 5. Box Plot
    st.subheader("Daily Return Distribution — Box Plot")

    fig_box = go.Figure()
    for tkr in show_tickers:
        r = price_data[tkr]["Daily Return"].dropna()
        fig_box.add_trace(go.Box(y=r, name=tkr, boxmean="sd"))
    fig_box.update_layout(
        title="Daily Return Distributions",
        yaxis_title="Daily Return",
        yaxis_tickformat=".1%",
        template="plotly_white",
        height=450,
        showlegend=False,
    )
    st.plotly_chart(fig_box, use_container_width=True, key="fig_box")


# ════════════════════════════════════════════════════════════════
# TAB 4 – Correlation & Portfolio
# ════════════════════════════════════════════════════════════════
with tab_corr:
    st.header("Correlation and Diversification")

    # Returns DataFrame for user stocks (no benchmark)
    returns_df = pd.concat(
        [price_data[t]["Daily Return"].rename(t) for t in selected_stocks], axis=1
    ).dropna()

    # 1. Correlation Heatmap
    st.subheader("Correlation Heatmap")
    st.caption("Pairwise Pearson correlation of daily returns. Diverging scale centered at 0.")

    corr_matrix = returns_df.corr()

    fig_heatmap = go.Figure(
        go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns.tolist(),
            y=corr_matrix.index.tolist(),
            colorscale="RdBu",
            zmid=0,
            zmin=-1,
            zmax=1,
            text=corr_matrix.round(2).values,
            texttemplate="%{text:.2f}",
            hoverongaps=False,
        )
    )
    fig_heatmap.update_layout(
        title="Pairwise Correlation Matrix of Daily Returns",
        template="plotly_white",
        height=400,
        xaxis_title="Ticker",
        yaxis_title="Ticker",
    )
    st.plotly_chart(fig_heatmap, use_container_width=True, key="fig_heatmap")

    st.divider()

    # 2. Scatter Plot
    st.subheader("Return Scatter Plot")

    col_sa, col_sb = st.columns(2)
    scatter_a = col_sa.selectbox("Stock A", options=selected_stocks, index=0, key="scatter_a")
    scatter_b = col_sb.selectbox(
        "Stock B", options=selected_stocks,
        index=min(1, len(selected_stocks) - 1), key="scatter_b",
    )

    if scatter_a == scatter_b:
        st.info("Select two different stocks to display the scatter plot.")
    else:
        scatter_data = returns_df[[scatter_a, scatter_b]].dropna()
        corr_val = scatter_data[scatter_a].corr(scatter_data[scatter_b])

        fig_scatter = go.Figure()
        fig_scatter.add_trace(
            go.Scatter(
                x=scatter_data[scatter_a], y=scatter_data[scatter_b],
                mode="markers",
                marker=dict(size=4, opacity=0.5, color="steelblue"),
                name="Daily Returns",
            )
        )
        fig_scatter.update_layout(
            title=f"{scatter_a} vs {scatter_b} Daily Returns  (r = {corr_val:.3f})",
            xaxis_title=f"{scatter_a} Daily Return",
            yaxis_title=f"{scatter_b} Daily Return",
            xaxis_tickformat=".1%",
            yaxis_tickformat=".1%",
            template="plotly_white",
            height=400,
        )
        st.plotly_chart(fig_scatter, use_container_width=True, key="fig_scatter")

    st.divider()

    # 3. Rolling Correlation
    st.subheader("Rolling Correlation")

    col_rca, col_rcb, col_rcw = st.columns(3)
    rc_a = col_rca.selectbox("Stock A", options=selected_stocks, index=0, key="rc_a")
    rc_b = col_rcb.selectbox(
        "Stock B", options=selected_stocks,
        index=min(1, len(selected_stocks) - 1), key="rc_b",
    )
    rc_window = col_rcw.select_slider(
        "Window (days)", options=[30, 60, 90], value=60, key="rc_window",
    )

    if rc_a == rc_b:
        st.info("Select two different stocks to display rolling correlation.")
    else:
        roll_corr = returns_df[rc_a].rolling(rc_window).corr(returns_df[rc_b])
        fig_rc = go.Figure()
        fig_rc.add_trace(
            go.Scatter(
                x=roll_corr.index, y=roll_corr, mode="lines",
                name=f"{rc_a} / {rc_b}",
                line=dict(color="steelblue", width=1.5),
            )
        )
        fig_rc.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="0")
        fig_rc.update_layout(
            title=f"{rc_window}-Day Rolling Correlation: {rc_a} vs {rc_b}",
            yaxis_title="Correlation",
            xaxis_title="Date",
            template="plotly_white",
            height=350,
        )
        st.plotly_chart(fig_rc, use_container_width=True, key="fig_rc")

    st.divider()

    # 4. Two-Asset Portfolio Explorer
    st.subheader("Two-Asset Portfolio Explorer")
    st.markdown(
        """
        **About diversification:** Combining two assets can produce a portfolio with *lower* volatility
        than either asset individually. This effect is strongest when the two assets have low or negative
        correlation. The curve below plots portfolio annualised volatility across every possible weight
        combination from 0 % to 100 %. When correlation is less than 1, the curve dips below a straight
        line connecting the two endpoints — meaning some blended portfolios achieve lower risk than
        either pure position. This is the core principle of diversification.
        """
    )

    col_pa, col_pb = st.columns(2)
    port_a = col_pa.selectbox("Stock A", options=selected_stocks, index=0, key="port_a")
    port_b = col_pb.selectbox(
        "Stock B", options=selected_stocks,
        index=min(1, len(selected_stocks) - 1), key="port_b",
    )

    if port_a == port_b:
        st.info("Select two different stocks to explore the portfolio.")
    else:
        weight_pct = st.slider(
            f"Weight in {port_a}  (remainder goes to {port_b})",
            min_value=0, max_value=100, value=50, step=1,
            format="%d%%", key="weight_slider",
        )
        w = weight_pct / 100.0

        aligned = returns_df[[port_a, port_b]].dropna()

        # Annualised parameters
        mu_a = aligned[port_a].mean() * 252
        mu_b = aligned[port_b].mean() * 252
        sigma_a = aligned[port_a].std() * math.sqrt(252)
        sigma_b = aligned[port_b].std() * math.sqrt(252)
        cov_ab = aligned.cov().loc[port_a, port_b] * 252  # annualised covariance

        # Current portfolio
        port_return = w * mu_a + (1 - w) * mu_b
        port_var = (
            w ** 2 * sigma_a ** 2
            + (1 - w) ** 2 * sigma_b ** 2
            + 2 * w * (1 - w) * cov_ab
        )
        port_vol = math.sqrt(max(port_var, 0))

        m1, m2, m3 = st.columns(3)
        m1.metric(f"Weight in {port_a}", f"{weight_pct}%")
        m2.metric("Portfolio Ann. Return", f"{port_return:.2%}")
        m3.metric("Portfolio Ann. Volatility", f"{port_vol:.2%}")

        # Full volatility curve
        weights = np.linspace(0, 1, 201)
        vols = np.sqrt(
            weights ** 2 * sigma_a ** 2
            + (1 - weights) ** 2 * sigma_b ** 2
            + 2 * weights * (1 - weights) * cov_ab
        )

        fig_port = go.Figure()
        fig_port.add_trace(
            go.Scatter(
                x=weights * 100, y=vols, mode="lines",
                name="Portfolio Volatility",
                line=dict(color="steelblue", width=2),
            )
        )
        fig_port.add_trace(
            go.Scatter(
                x=[weight_pct], y=[port_vol], mode="markers",
                name=f"Current ({weight_pct}% {port_a})",
                marker=dict(color="crimson", size=12, symbol="diamond"),
            )
        )
        fig_port.add_trace(
            go.Scatter(
                x=[0, 100], y=[sigma_b, sigma_a], mode="markers",
                name="Individual stocks",
                marker=dict(color="gray", size=10, symbol="circle"),
            )
        )
        fig_port.update_layout(
            title=f"Portfolio Volatility vs. Weight in {port_a}",
            xaxis_title=f"Weight in {port_a} (%)",
            yaxis_title="Annualised Volatility",
            yaxis_tickformat=".1%",
            template="plotly_white",
            height=420,
        )
        st.plotly_chart(fig_port, use_container_width=True, key="fig_port")


# ════════════════════════════════════════════════════════════════
# TAB 5 – About
# ════════════════════════════════════════════════════════════════
with tab_about:
    st.header("About & Methodology")
    st.markdown(
        """
        ### What this app does
        This dashboard downloads historical adjusted closing prices for 2–5 user-selected stocks and
        the S&P 500 benchmark (^GSPC) via `yfinance`, then provides four analytical sections:

        | Tab | Contents |
        |---|---|
        | **Overview** | Key metrics, closing price chart, normalised returns |
        | **Price & Returns** | Adjusted price chart, summary statistics, cumulative wealth index |
        | **Risk & Distribution** | Rolling volatility, histogram, Q-Q plot, Jarque-Bera test, box plot |
        | **Correlation & Portfolio** | Correlation heatmap, return scatter, rolling correlation, two-asset explorer |

        ### Key assumptions
        | Item | Convention |
        |---|---|
        | Trading days per year | 252 |
        | Return type | Simple (arithmetic) — `pct_change()` |
        | Cumulative wealth | `(1 + r).cumprod()` starting from $10,000 |
        | Equal-weight portfolio | Daily average of selected stocks' simple returns |
        | Missing data threshold | > 5 % missing → ticker dropped with a warning |
        | Data alignment | Truncated to overlapping date range; remaining gaps forward-filled |

        ### Formulas
        - **Annualised return:** `mean(daily r) × 252`
        - **Annualised volatility:** `std(daily r) × √252`
        - **Two-asset portfolio variance:** `w²σ²A + (1−w)²σ²B + 2w(1−w)σAB`
          where σAB is the annualised covariance between A and B.
        - **Jarque-Bera test:** tests whether skewness and excess kurtosis match a normal distribution;
          p < 0.05 rejects normality at the 5 % significance level.

        ### Data source
        All prices are sourced from Yahoo Finance via the `yfinance` library.
        Adjusted close prices account for dividends and stock splits.

        ### Caching
        Downloads and computations are cached for 1 hour with `@st.cache_data`
        to avoid unnecessary re-fetches on widget interactions.
        """
    )
