# Quant Stock Analytics Dashboard

An interactive quantitative finance dashboard built with Python and Streamlit to analyze stock prices, risk metrics, benchmark performance, and predictive signals[cite: 1].

## 🌟 Key Features
* **Risk & Performance Metrics:** Calculates annualized volatility, Sharpe ratio, and 95% Daily Value at Risk (VaR)[cite: 1].
* **Benchmark Analytics:** Evaluates stock Beta and correlation against the NIFTY 50 index[cite: 1].
* **Relative Performance:** Features base-100 rebased return charts for fair multi-stock comparison[cite: 1].
* **Machine Learning Experiment:** Baseline directional return classification using historical technical features[cite: 1].

## 🛠️ Tech Stack
* **Frontend/UI:** Streamlit
* **Data Processing:** Pandas, NumPy
* **Visualization:** Plotly
* **Machine Learning:** Scikit-Learn
* **Data Source:** Yahoo Finance (`yfinance`)

## 🚀 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/scientifi2412/quant-stock-analytics.git](https://github.com/scientifi2412/quant-stock-analytics.git)
   cd quant-stock-analytics
   
2.install dependencies 
 pip install -r requirements.txt  
3. launch the app
streamlit run app.py
4. streamlit run app.py
