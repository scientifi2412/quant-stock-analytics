# 📊 Quant Stock Analytics & Predictive Signal Platform

**An interactive quantitative finance dashboard for analyzing stock performance, risk, market relationships, and machine-learning-based directional signals.**

Built with **Python, Streamlit, Pandas, Plotly, Scikit-learn, and Yahoo Finance**, this application brings quantitative finance concepts into an interactive dashboard where users can explore real market data and evaluate stocks from multiple perspectives.


## Table of Contents

- Overview
- Features
- Quantitative Concepts Used
- Technology Stack
- Project Structure
- Getting Started
- Why I Built This
- Future Improvements
- Disclaimer


---

## Overview

**Quant Stock Analytics** combines several disciplines into a single dashboard:

**Financial Markets + Statistics + Risk Analytics + Data Visualization + Machine Learning**

Instead of examining metrics in isolation, the app brings them together so users can explore the behavior and risk characteristics of different stocks side by side.

---

## Features

### 📈 Analyze Stock Performance
Select a stock and explore its historical price and return behavior:
- Historical stock prices
- Daily returns
- Cumulative returns
- Annualized performance
- Interactive price and return charts

### ⚠️ Measure Investment Risk
The app calculates key quantitative risk metrics:

| Metric | What It Tells You |
|---|---|
| **Volatility** | How much the stock's returns fluctuate |
| **Sharpe Ratio** | Risk-adjusted performance |
| **95% VaR** | Potential daily loss at a 95% confidence level |
| **Maximum Drawdown** | Largest historical decline |

These metrics let users evaluate both **return and risk**, rather than price performance alone.

### 📊 Compare Against the NIFTY 50
Using the NIFTY 50 as a market benchmark, the app calculates:
- **Beta**
- **Correlation**
- Stock vs. benchmark returns
- Relative market sensitivity

Helps answer: *"How sensitive is this stock to overall market movements?"*

### 🔄 Compare Multiple Stocks
Using **Base-100 rebased performance**, each selected stock is normalized to a common starting point (100), making it easy to visually compare assets with different original share prices over the same period.

### 🤖 Machine Learning Prediction
A **Logistic Regression** model estimates the probability of the next trading day's direction, using historical market features. The output is a probability-based signal rather than a simple binary prediction — positioning the ML component as a **quantitative research experiment**, not a guaranteed trading strategy.

---

## Quantitative Concepts Used

**Risk**
- Volatility
- Sharpe Ratio
- Value at Risk (VaR)
- Drawdown

**Market Analysis**
- Beta
- Correlation
- Benchmark analysis
- Relative performance

**Statistics**
- Return distributions
- Historical returns
- Rolling statistics
- Normalization

**Machine Learning**
- Feature engineering
- Binary classification
- Logistic Regression
- Probability estimation

---

## Technology Stack

| Area | Technology |
|---|---|
| Programming | Python |
| Dashboard | Streamlit |
| Data Analysis | Pandas |
| Numerical Computing | NumPy |
| Visualization | Plotly |
| Machine Learning | Scikit-learn |
| Financial Data | Yahoo Finance / yfinance |

---

## Project Structure

```text
quant-stock-analytics/
│
├── app.py
├── requirements.txt
├── README.md

```

---

## Getting Started

### 1. Clone the repository
```bash
https://github.com/scientifi2412/quant-stock-analytics.git
cd quant-stock-analytics
```

### 2. Install dependencies
```bash
pip install - requirements.txt
```

### 3. Run the app
```bash
streamlit run app.py
```

The dashboard will open automatically in your browser.

---

## Why I Built This

This project explores how **quantitative methods can be applied to real-world financial market data**, combining ideas from statistics, financial markets, risk management, data analysis, and machine learning into a single interactive application. It also serves as a foundation for more advanced quantitative research projects.

---

## Disclaimer

This application is intended for **educational and quantitative research purposes only**. The machine-learning predictions and financial metrics should not be interpreted as investment advice or guarantees of future market performance.
