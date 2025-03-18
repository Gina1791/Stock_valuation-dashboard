import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import matplotlib.pyplot as plt

# Streamlit App Title
st.title("📊 Stock Valuation & Prediction Dashboard")

# Sidebar - User selects stocks
tickers = ["AAPL", "MSFT", "GOOGL", "NVDA", "MU", "AMZN", "META", "V", "MA", "GS", "DB", "TSLA",
           "BRK-B", "JPM", "JNJ", "WMT", "PG", "UNH", "BAC", "XOM", "DIS", "HD", "CVX", "KO", "PFE",
           "MRK", "INTC", "VZ", "CSCO", "CRM"]
selected_ticker = st.sidebar.selectbox("Select a stock:", tickers)

start_date = "2022-01-01"
end_date = "2025-02-24"

@st.cache_data
def fetch_stock_data(ticker):
    """Fetch stock historical data"""
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

data = fetch_stock_data(selected_ticker)

# Display Stock Data
st.subheader(f"📈 {selected_ticker} Stock Data")
st.dataframe(data.tail(10))

# Plot Stock Price
st.subheader("📊 Stock Closing Price Trend")
st.line_chart(data['Close'])

# Fetch Financial Metrics
@st.cache_data
def get_financials(ticker):
    try:
        stock = yf.Ticker(ticker)
        income_stmt = stock.financials
        balance_sheet = stock.balance_sheet
        valuation_metrics = stock.info
        cash_flow = stock.cashflow

        metrics = {
            'P/E Ratio': valuation_metrics.get('trailingPE', np.nan),
            'P/B Ratio': valuation_metrics.get('priceToBook', np.nan),
            'Debt-to-Equity': valuation_metrics.get('debtToEquity', np.nan),
            'Revenue Growth Rate (YoY)': ((income_stmt.loc['Total Revenue'].iloc[0] - income_stmt.loc['Total Revenue'].iloc[1]) / income_stmt.loc['Total Revenue'].iloc[1]) * 100 
                if 'Total Revenue' in income_stmt and len(income_stmt.loc['Total Revenue']) >= 2 else np.nan,
            'ROE': (income_stmt.loc['Net Income'].iloc[0] / balance_sheet.loc['Stockholders Equity'].iloc[0]) * 100 
                if 'Stockholders Equity' in balance_sheet and 'Net Income' in income_stmt else np.nan,
            'ROA': (income_stmt.loc['Net Income'].iloc[0] / balance_sheet.loc['Total Assets'].iloc[0]) * 100 
                if 'Total Assets' in balance_sheet and 'Net Income' in income_stmt else np.nan,
            'FCF Yield': (cash_flow.loc['Free Cash Flow'].iloc[0] / valuation_metrics.get('marketCap', np.nan)) * 100 
                if 'Free Cash Flow' in cash_flow and 'marketCap' in valuation_metrics else np.nan
        }
        
        # Filter out NaN values
        filtered_metrics = {k: v for k, v in metrics.items() if not np.isnan(v)}
        return filtered_metrics
    except Exception as e:
        return {"Error": str(e)}

financials = get_financials(selected_ticker)

st.subheader("📊 Financial Metrics")
st.json(financials)

# Determine if Stock is Overvalued or Undervalued
if 'P/E Ratio' in financials and 'P/B Ratio' in financials:
    pe_ratio = financials['P/E Ratio']
    pb_ratio = financials['P/B Ratio']
    
    if pe_ratio < 15 and pb_ratio < 1.5:
        valuation_status = "Undervalued ✅"
    elif pe_ratio > 25 or pb_ratio > 3:
        valuation_status = "Overvalued ❌"
    else:
        valuation_status = "Fairly Valued ⚖️"
    
    st.subheader("💰 Stock Valuation Status")
    st.write(f"**{selected_ticker} is {valuation_status}**")
