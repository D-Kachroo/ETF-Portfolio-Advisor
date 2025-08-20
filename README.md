# 📊 Full-Stack Quant ETF Investment Portfolio Advisor

**Developer:** David Kachroo (CFM @ University of Waterloo)

A full-stack ETF (Exchange-Traded Funds) portfolio advisory and optimization tool that is powered by real-time Yahoo Finance data, AI, machine learning (ML), and various Python libraries. Developed on VS Code with a Streamlit frontend.

## 🔧 App Features

- A multi-sector ETF selection bar with cleaned metadata
- Adjusted 'Close' price data from Yahoo Finance
- Risk metrics: volatility, maximum drawdown, CVaR (95%), Sharpe ratio
- Portfolio optimization via Efficient Frontier with L2 regularization
- 30-day return prediction using ML algorithms (Random Forest)
- Correlation heatmap of ETF returns (Pearson's Coefficient)
- Integration of the FRED API for CPI, interest rates, and U.S. unemployment
- Exportable price data (Excel/CSV file)
- Streamlit UI with error handling and input validation

## ⚙️ Tech Stack

`Python` · `pandas` · `NumPy` · `yfinance` · `FRED API` · `PyPortfolioOpt` · `scikit-learn` · `Random Forest (ML)` · `Matplotlib` · `Streamlit` · `Docker` · `Railway`

## 🚀 Streamlit Link

```bash
https://quant-etf-advisor.streamlit.app/
