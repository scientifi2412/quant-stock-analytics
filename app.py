import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

st.set_page_config(
    page_title="Quant Stock Analytics",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Quant Stock Analytics")

st.markdown(
    "A quantitative dashboard for analyzing stock performance, "
    "returns, volatility, risk, and market relationships."
)

st.sidebar.header("⚙️ Dashboard Settings")

market = st.sidebar.selectbox(
    "🌎 Select Market",
    ["🇮🇳 Indian Stocks", "🇺🇸 US Stocks"]
)

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

    benchmark = "^NSEI"
    benchmark_name = "NIFTY 50"

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

    benchmark = "^GSPC"
    benchmark_name = "S&P 500"

company = st.sidebar.selectbox(
    "📊 Select Stock",
    list(stocks.keys())
)

ticker = stocks[company]

period = st.sidebar.selectbox(
    "📅 Historical Period",
    ["6mo", "1y", "2y", "5y"],
    index=1
)

st.sidebar.markdown("---")

st.sidebar.info(
    f"Analyzing: **{company}**\n\n"
    f"Ticker: **{ticker}**\n\n"
    f"Benchmark: **{benchmark_name}**"
)


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
    market_data = get_stock_data(benchmark, period)

    if data.empty:
        st.error(f"No data found for {company}.")
        st.stop()

    if market_data.empty:
        st.error(f"No benchmark data found for {benchmark_name}.")
        st.stop()

    data["Daily Return"] = data["Close"].pct_change()

    data["20 Day MA"] = (
        data["Close"].rolling(20).mean()
    )

    data["50 Day MA"] = (
        data["Close"].rolling(50).mean()
    )

    data["Cumulative Return"] = (
        (1 + data["Daily Return"].fillna(0)).cumprod() - 1
    )

    data["Rolling Volatility"] = (
        data["Daily Return"]
        .rolling(20)
        .std()
        * np.sqrt(252)
    )

    market_data["Market Return"] = (
        market_data["Close"].pct_change()
    )

    combined_returns = pd.concat(
        [
            data["Daily Return"],
            market_data["Market Return"]
        ],
        axis=1,
        join="inner"
    )

    combined_returns.columns = [
        "Stock Return",
        "Market Return"
    ]

    combined_returns = combined_returns.dropna()

    annualized_volatility = (
        data["Daily Return"].std() * np.sqrt(252)
    )

    annualized_return = (
        data["Daily Return"].mean() * 252
    )

    risk_free_rate = 0.04

    if annualized_volatility != 0:
        sharpe_ratio = (
            annualized_return - risk_free_rate
        ) / annualized_volatility
    else:
        sharpe_ratio = 0

    cumulative_max = data["Close"].cummax()

    drawdown = (
        data["Close"] - cumulative_max
    ) / cumulative_max

    max_drawdown = drawdown.min()

    market_variance = (
        combined_returns["Market Return"].var()
    )

    if market_variance != 0:
        beta = (
            combined_returns["Stock Return"]
            .cov(combined_returns["Market Return"])
            / market_variance
        )
    else:
        beta = 0

    correlation = (
        combined_returns["Stock Return"]
        .corr(combined_returns["Market Return"])
    )

    var_95 = (
        data["Daily Return"].quantile(0.05)
    )

    latest_price = float(
        data["Close"].iloc[-1]
    )

    latest_return = float(
        data["Daily Return"].iloc[-1]
    )

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
        f"{annualized_volatility:.2%}"
    )

    col4.metric(
        "Maximum Drawdown",
        f"{max_drawdown:.2%}"
    )

    st.subheader("📈 Stock Price")

    st.line_chart(data[["Close"]])

    st.subheader("📉 Moving Averages")

    moving_average_data = data[
        ["Close", "20 Day MA", "50 Day MA"]
    ].dropna()

    st.line_chart(moving_average_data)

    st.subheader("📈 Cumulative Returns")

    st.line_chart(data["Cumulative Return"])

    st.subheader("📊 Daily Returns")

    st.line_chart(
        data["Daily Return"].dropna()
    )

    st.subheader("📉 20-Day Rolling Volatility")

    st.line_chart(
        data["Rolling Volatility"].dropna()
    )

    st.subheader("📊 Return Distribution")

    returns = data["Daily Return"].dropna()

    histogram_counts, histogram_edges = np.histogram(
        returns,
        bins=30
    )

    histogram_df = pd.DataFrame({
        "Return Range": [
            f"{histogram_edges[i]:.2%} to "
            f"{histogram_edges[i + 1]:.2%}"
            for i in range(len(histogram_counts))
        ],
        "Number of Days": histogram_counts
    })

    st.bar_chart(
        histogram_df.set_index("Return Range")
    )

    st.subheader("⚠️ Quantitative Risk Metrics")

    risk_col1, risk_col2, risk_col3 = st.columns(3)

    risk_col1.metric(
        "Sharpe Ratio",
        f"{sharpe_ratio:.2f}"
    )

    risk_col2.metric(
        "Beta",
        f"{beta:.2f}"
    )

    risk_col3.metric(
        "95% Daily VaR",
        f"{var_95:.2%}"
    )

    st.subheader(
        f"🔗 Relationship with {benchmark_name}"
    )

    relationship_col1, relationship_col2 = st.columns(2)

    relationship_col1.metric(
        "Correlation",
        f"{correlation:.2f}"
    )

    relationship_col2.metric(
        "Beta",
        f"{beta:.2f}"
    )

    st.markdown("---")

    st.header("📊 Compare Stocks")

    st.markdown(
        "Compare multiple stocks based on performance, "
        "volatility, Sharpe ratio, and maximum drawdown."
    )

    comparison_names = st.multiselect(
        "Select stocks to compare",
        list(stocks.keys()),
        default=list(stocks.keys())[:3],
        max_selections=5
    )

    if len(comparison_names) < 2:
        st.info(
            "Please select at least 2 stocks to compare."
        )

    else:
        comparison_data = {}
        comparison_metrics = []

        for stock_name in comparison_names:

            stock_symbol = stocks[stock_name]

            stock_data = get_stock_data(
                stock_symbol,
                period
            )

            if stock_data.empty:
                continue

            stock_data["Return"] = (
                stock_data["Close"].pct_change()
            )

            normalized_price = (
                stock_data["Close"]
                / stock_data["Close"].iloc[0]
                * 100
            )

            comparison_data[stock_name] = (
                normalized_price
            )

            stock_return = (
                stock_data["Return"].dropna()
            )

            stock_volatility = (
                stock_return.std() * np.sqrt(252)
            )

            stock_annual_return = (
                stock_return.mean() * 252
            )

            if stock_volatility != 0:
                stock_sharpe = (
                    stock_annual_return
                    - risk_free_rate
                ) / stock_volatility
            else:
                stock_sharpe = 0

            stock_cumulative_max = (
                stock_data["Close"].cummax()
            )

            stock_drawdown = (
                stock_data["Close"]
                - stock_cumulative_max
            ) / stock_cumulative_max

            stock_max_drawdown = (
                stock_drawdown.min()
            )

            comparison_metrics.append({
                "Stock": stock_name,
                "Annualized Return":
                    stock_annual_return,
                "Volatility":
                    stock_volatility,
                "Sharpe Ratio":
                    stock_sharpe,
                "Maximum Drawdown":
                    stock_max_drawdown
            })

        if comparison_data:

            comparison_chart = pd.DataFrame(
                comparison_data
            )

            st.subheader(
                "📈 Relative Performance"
            )

            st.caption(
                "Each stock starts at 100 so that "
                "percentage performance can be compared fairly."
            )

            st.line_chart(
                comparison_chart
            )

        if comparison_metrics:

            metrics_df = pd.DataFrame(
                comparison_metrics
            )

            st.subheader(
                "📋 Comparison Metrics"
            )

            display_df = metrics_df.copy()

            display_df["Annualized Return"] = (
                display_df["Annualized Return"]
                .map(lambda x: f"{x:.2%}")
            )

            display_df["Volatility"] = (
                display_df["Volatility"]
                .map(lambda x: f"{x:.2%}")
            )

            display_df["Sharpe Ratio"] = (
                display_df["Sharpe Ratio"]
                .map(lambda x: f"{x:.2f}")
            )

            display_df["Maximum Drawdown"] = (
                display_df["Maximum Drawdown"]
                .map(lambda x: f"{x:.2%}")
            )

            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True
            )

        if len(comparison_data) >= 2:

            st.subheader(
                "🔗 Stock Return Correlation"
            )

            comparison_prices = pd.DataFrame(
                comparison_data
            )

            comparison_returns = (
                comparison_prices
                .pct_change()
                .dropna()
            )

            correlation_matrix = (
                comparison_returns.corr()
            )

            st.dataframe(
                correlation_matrix.style.format(
                    "{:.2f}"
                ),
                use_container_width=True
            )

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
