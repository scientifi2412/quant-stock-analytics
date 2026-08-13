import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score

st.set_page_config(
    page_title="Quant Stock Analytics",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Quant Stock Analytics")

st.markdown(
    "A quantitative dashboard for analyzing stock prices, "
    "returns, risk, stock comparisons, and machine learning signals."
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

    st.line_chart(
        data[["Close"]]
    )

    st.subheader("📉 Moving Averages")

    moving_average_data = data[
        ["Close", "20 Day MA", "50 Day MA"]
    ].dropna()

    st.line_chart(
        moving_average_data
    )

    st.subheader("📈 Cumulative Returns")

    st.line_chart(
        data["Cumulative Return"]
    )

    st.subheader("📊 Daily Returns")

    st.line_chart(
        data["Daily Return"].dropna()
    )

    st.subheader("📉 20-Day Rolling Volatility")

    st.line_chart(
        data["Rolling Volatility"].dropna()
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

            comparison_data[stock_name] = normalized_price

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

            stock_max_drawdown = stock_drawdown.min()

            comparison_metrics.append({
                "Stock": stock_name,
                "Annualized Return": stock_annual_return,
                "Volatility": stock_volatility,
                "Sharpe Ratio": stock_sharpe,
                "Maximum Drawdown": stock_max_drawdown
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

    st.markdown("---")

    st.header("🤖 Machine Learning Analysis")

    st.markdown(
        "A simple Logistic Regression model that estimates "
        "whether the next trading day's return will be positive."
    )

    ml_data = data.copy()

    ml_data["5 Day Return"] = (
        ml_data["Close"].pct_change(5)
    )

    ml_data["20 Day Return"] = (
        ml_data["Close"].pct_change(20)
    )

    ml_data["Volume Change"] = (
        ml_data["Volume"].pct_change()
    )

    ml_data["Next Day Return"] = (
        ml_data["Close"].shift(-1)
        / ml_data["Close"]
        - 1
    )

    ml_data["Target"] = (
        ml_data["Next Day Return"] > 0
    ).astype(int)

    feature_columns = [
        "Daily Return",
        "5 Day Return",
        "20 Day Return",
        "20 Day MA",
        "50 Day MA",
        "Rolling Volatility",
        "Volume Change"
    ]

    # Remove infinite values created by percentage changes.
    ml_data = ml_data.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Remove rows with missing values.
    ml_data = ml_data.dropna(
        subset=feature_columns + ["Target"]
    )

    if len(ml_data) < 100:

        st.warning(
            "Not enough historical observations for "
            "a meaningful machine learning experiment."
        )

    else:

        X = ml_data[feature_columns]
        y = ml_data["Target"]

        split_index = int(
            len(ml_data) * 0.8
        )

        X_train = X.iloc[:split_index]
        X_test = X.iloc[split_index:]

        y_train = y.iloc[:split_index]
        y_test = y.iloc[split_index:]

        model = LogisticRegression(
            max_iter=1000
        )

        model.fit(
            X_train,
            y_train
        )

        predictions = model.predict(
            X_test
        )

        probabilities = model.predict_proba(
            X_test
        )[:, 1]

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        precision = precision_score(
            y_test,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            predictions,
            zero_division=0
        )

        ml_col1, ml_col2, ml_col3 = st.columns(3)

        ml_col1.metric(
            "Accuracy",
            f"{accuracy:.2%}"
        )

        ml_col2.metric(
            "Precision",
            f"{precision:.2%}"
        )

        ml_col3.metric(
            "Recall",
            f"{recall:.2%}"
        )

        st.subheader(
            "📈 ML Predictions"
        )

        prediction_data = pd.DataFrame(
            {
                "Probability of Positive Return":
                    probabilities
            },
            index=X_test.index
        )

        st.line_chart(
            prediction_data
        )

        st.subheader(
            "🧠 Latest Model Signal"
        )

        latest_features = X.iloc[[-1]]

        latest_prediction = model.predict(
            latest_features
        )[0]

        latest_probability = model.predict_proba(
            latest_features
        )[0][1]

        if latest_prediction == 1:

            st.success(
                f"Model signal: Positive return\n\n"
                f"Estimated probability of a positive "
                f"next-day return: {latest_probability:.2%}"
            )

        else:

            st.warning(
                f"Model signal: Negative return\n\n"
                f"Estimated probability of a positive "
                f"next-day return: {latest_probability:.2%}"
            )

        st.caption(
            "This is a historical machine-learning experiment, "
            "not a guaranteed prediction or investment recommendation."
        )

        st.subheader(
            "🔍 ML Features"
        )

        feature_description = pd.DataFrame(
            {
                "Feature": [
                    "Daily Return",
                    "5 Day Return",
                    "20 Day Return",
                    "20 Day MA",
                    "50 Day MA",
                    "20 Day Volatility",
                    "Volume Change"
                ],
                "Meaning": [
                    "Today's percentage price change",
                    "Price change over the last 5 trading days",
                    "Price change over the last 20 trading days",
                    "Average closing price over 20 days",
                    "Average closing price over 50 days",
                    "Recent return variability",
                    "Percentage change in trading volume"
                ]
            }
        )

        st.dataframe(
            feature_description,
            use_container_width=True,
            hide_index=True
        )

    with st.expander("🔎 View Historical Data"):

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
