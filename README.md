# 📊 Full-Stack Quant ETF Investment Portfolio Advisor

**Creator:** David Kachroo (CFM @ University of Waterloo)

A full-stack ETF portfolio advisory and optimization tool that is powered by real-time financial ETF data, machine learning (ML), and global economic indicators. Built in Python on VS Code with a Streamlit frontend.

## 🔧 App Features

- Sector-based ETF selection with cleaned metadata
- Adjusted 'Close' price data from Yahoo Finance
- Risk metrics: volatility, max drawdown, CVaR (95%), Sharpe ratio
- Portfolio optimization via Efficient Frontier with L2 regularization
- 30-day return prediction using ML algorithms (Random Forest)
- Correlation heatmap of ETF returns (Pearson's Coefficient)
- FRED API integration: CPI, interest rate, unemployment
- Exportable price data (Excel/CSV file)
- Streamlit UI with error handling and input validation

## ⚙️ Tech Stack

`Python` · `Streamlit` · `yfinance` · `PyPortfolioOpt` · `pandas` · `FRED API` · `scikit-learn`

## 🚀 Run Locally

```bash
streamlit run ETF_project.py
