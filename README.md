# 📊 Full-Stack Quantitative ETF Investment Portfolio Robo-Advisor

**Author:** David Kachroo (CFM @ University of Waterloo)

A full-stack ETF portfolio advisory and optimization tool powered by real-time financial data, machine learning, and economic indicators. Built in Python with a Streamlit frontend.

## 🔧 App Features

- Sector-based ETF selection with cleaned metadata
- Adjusted price retrieval from Yahoo Finance and return calculation
- Risk metrics: volatility, max drawdown, CVaR (95%), Sharpe ratio
- Portfolio optimization via Efficient Frontier with L2 regularization
- 30-day return prediction using ML/Random Forest and global economic data
- Correlation heatmap of ETF returns (Pearson's)
- FRED API integration: CPI, interest rate, unemployment
- Exportable price data (Excel/CSV file)
- Streamlit UI with error handling and input validation

## ⚙️ Tech Stack

`Python` · `Streamlit` · `yfinance` · `PyPortfolioOpt` · `pandas` · `FRED API` · `scikit-learn`

## 🚀 Run Locally

```bash
streamlit run ETF_project.py
