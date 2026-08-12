import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

# -----------------------------
# PAGE CONFIGURATION
# -----------------------------

st.set_page_config(
    page_title="Quant Stock Analytics",
    page_icon="📈",
    layout="wide"
)

# -----------------------------
# TITLE
# -----------------------------

st.title("📈 Quant Stock Analytics")

st.markdown(
    "A quantitative dashboard for analyzing stock prices, "
    "returns, moving averages, and risk."
)

# -----------------------------
# SIDEBAR
# -----------------------------

st.sidebar.header("⚙️ Dashboard Settings")

# Select market
market = st.sidebar.selectbox(
    "🌎 Select Market",
    ["🇮🇳 Indian Stocks", "🇺🇸 US Stocks"]
)

# Indian stocks
if market == "🇮🇳 Indian Stocks":

    stocks = {
        "Reliance Industries": "RELIANCE.NS",
        "Tata Consultancy Services": "TCS.NS",
        "Infosys": "INFY.NS",
        "HDFC Bank": "HDFCBANK.NS",
        "ICICI Bank": "ICICIBANK.NS",
        "State Bank of India": "SBIN.NS",
        "Tata Motors": "TATAMOTORS.NS",
        "Larsen & Toubro": "LT.NS",
        "Bharti Airtel": "BHARTIARTL.NS",
        "ITC": "ITC.NS"
    }

# US stocks
else:

    stocks = {
        "Apple": "AAPL",
        "Microsoft": "MSFT",
        "NVIDIA": "NVDA",
        "Amazon": "AMZN",
        "Alphabet (Google)": "GOOGL",
        "Meta": "META",
        "Tesla": "TSLA",
        "JPMorgan Chase": "JPM"
    }

# Select company
company = st.sidebar.selectbox(
    "📊 Select Stock",
    list(stocks.keys())
)

# Automatically get ticker
ticker = stocks[company]

# Historical period
period = st.sidebar.selectbox(
    "📅 Historical Period",
    ["6mo", "1y", "2y", "5y"],
    index=1
)

st.sidebar.markdown("---")

st.sidebar.info(
    f"Analyzing: **{company}**\n\n"
    f"Ticker: **{ticker}**"
)

# -----------------------------
# DOWNLOAD DATA
# -----------------------------

@st.cache_data
def get_stock_data(symbol, selected_period):

    data = yf.download(
        symbol,
        period=selected_period,
        auto_adjust=True,
        progress=False
    )

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    return data


try:

    data = get_stock_data(ticker, period)

    if data.empty:

        st.error(
            f"No data found for '{ticker}'. "
            "Please select another stock."
        )

        st.stop()

    # -----------------------------
    # CALCULATE QUANT METRICS
    # -----------------------------

    data["Daily Return"] = data["Close"].pct_change()

    data["20 Day MA"] = (
        data["Close"].rolling(20).mean()
    )

    data["50 Day MA"] = (
        data["Close"].rolling(50).mean()
    )

    # Annualized volatility
    volatility = (
        data["Daily Return"].std()
        * np.sqrt(252)
    )

    # Annualized return
    mean_return = (
        data["Daily Return"].mean()
        * 252
    )

    # Sharpe ratio
    if volatility != 0:

        sharpe_ratio = (
            mean_return / volatility
        )

    else:

        sharpe_ratio = 0

    # Maximum drawdown
    cumulative_max = (
        data["Close"].cummax()
    )

    drawdown = (
        data["Close"] - cumulative_max
    ) / cumulative_max

    max_drawdown = drawdown.min()

    # Latest values
    latest_price = float(
        data["Close"].iloc[-1]
    )

    latest_return = float(
        data["Daily Return"].iloc[-1]
    )

    # -----------------------------
    # MARKET OVERVIEW
    # -----------------------------

    st.subheader(
        f"📊 {company} Market Overview"
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Latest Price",
        f"{latest_price:,.2f}"
    )

    col2.metric(
        "Daily Return",
        f"{latest_return:.2%}"
    )

    col3.metric(
        "Annualized Volatility",
        f"{volatility:.2%}"
    )

    col4.metric(
        "Maximum Drawdown",
        f"{max_drawdown:.2%}"
    )

    # -----------------------------
    # STOCK PRICE CHART
    # -----------------------------

    st.subheader("📈 Stock Price")

    chart_data = data[["Close"]].copy()

    st.line_chart(chart_data)

    # -----------------------------
    # MOVING AVERAGES
    # -----------------------------

    st.subheader("📉 Moving Averages")

    moving_average_data = data[
        ["Close", "20 Day MA", "50 Day MA"]
    ].dropna()

    st.line_chart(
        moving_average_data
    )

    # -----------------------------
    # DAILY RETURNS
    # -----------------------------

    st.subheader("📊 Daily Returns")

    st.line_chart(
        data["Daily Return"].dropna()
    )

    # -----------------------------
    # RISK METRICS
    # -----------------------------

    st.subheader(
        "⚠️ Quantitative Risk Metrics"
    )

    risk_col1, risk_col2, risk_col3 = st.columns(3)

    risk_col1.metric(
        "Annualized Volatility",
        f"{volatility:.2%}"
    )

    risk_col2.metric(
        "Sharpe Ratio",
        f"{sharpe_ratio:.2f}"
    )

    risk_col3.metric(
        "Maximum Drawdown",
        f"{max_drawdown:.2%}"
    )

    # -----------------------------
    # HISTORICAL DATA
    # -----------------------------

    with st.expander(
        "🔎 View Historical Data"
    ):

        st.dataframe(
            data.tail(50),
            use_container_width=True
        )

except Exception as e:

    st.error(
        "Something went wrong while loading "
        "the stock data."
    )

    st.exception(e)
