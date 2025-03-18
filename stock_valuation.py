
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

tickers = ["AAPL", "MSFT", "GOOGL", "NVDA", "MU", "AMZN", "META", "V", "MA", "GS", "DB", "TSLA",
            "BRK-B", "JPM", "JNJ", "WMT", "PG", "UNH", "BAC", "XOM", "DIS", "HD", "CVX", "KO", "PFE",
              "MRK", "INTC", "VZ", "CSCO", "CRM"]
start_date = "2022-01-01"
end_date = "2025-02-24"

# Download historical stock data
data = yf.download(tickers, start=start_date, end=end_date)
print(data.head())  # Display the first few rows of the data

# Define a function to classify the company
def stock_valuation(pe_ratio, pb_ratio, fcf_yield):
    if (pe_ratio > 25 and pb_ratio > 3) or (fcf_yield < 2):  # Example threshold
        return 1  # Overvalued
    return 0  # Undervalued

# Initialize an empty list to store financial data
financials_list = []

# Loop through each ticker
for ticker in tickers:
    try:
        # Fetch stock data
        stock = yf.Ticker(ticker)
        
        # Fetch financial statements and valuation metrics
        income_stmt = stock.financials
        balance_sheet = stock.balance_sheet
        valuation_metrics = stock.info
        cash_flow = stock.cashflow

        # Calculate ROE (Return on Equity)
        roe = np.nan
        if 'Stockholders Equity' in balance_sheet.index and 'Net Income' in income_stmt.index:
            total_equity = balance_sheet.loc['Stockholders Equity'].iloc[0]
            net_income = income_stmt.loc['Net Income'].iloc[0]
            if total_equity != 0:  # Avoid division by zero
                roe = (net_income / total_equity) * 100

        # Calculate ROA (Return on Assets)
        roa = np.nan
        if 'Total Assets' in balance_sheet.index and 'Net Income' in income_stmt.index:
            total_assets = balance_sheet.loc['Total Assets'].iloc[0]
            net_income = income_stmt.loc['Net Income'].iloc[0]
            if total_assets != 0:  # Avoid division by zero
                roa = (net_income / total_assets) * 100

        # Calculate Revenue Growth Rate (YoY)
        revenue_growth_yoy = np.nan
        if 'Total Revenue' in income_stmt.index and len(income_stmt.loc['Total Revenue']) >= 2:
            revenue_current_year = income_stmt.loc['Total Revenue'].iloc[0]  # Most recent year
            revenue_previous_year = income_stmt.loc['Total Revenue'].iloc[1]  # Previous year
            if revenue_previous_year != 0:
                revenue_growth_yoy = ((revenue_current_year - revenue_previous_year) / revenue_previous_year) * 100

        # Calculate Free Cash Flow (FCF)
        fcf_yield = np.nan
        if 'Free Cash Flow' in cash_flow.index and 'marketCap' in valuation_metrics.keys():
            market_cap = valuation_metrics['marketCap']
            fcf = cash_flow.loc['Free Cash Flow'].iloc[0]
            if market_cap != 0:  # Avoid division by zero
                fcf_yield = (fcf / market_cap) * 100

        # Get valuation metrics
        pe_ratio = valuation_metrics.get('trailingPE', np.nan)
        pb_ratio = valuation_metrics.get('priceToBook', np.nan)
        debt_to_equity = valuation_metrics.get('debtToEquity', np.nan)

        # Feature Engineering
        earnings_yield = np.nan if pe_ratio in [np.nan, 0] else (1 / pe_ratio)
        debt_service_ratio = np.nan if roe in [np.nan, 0] else (debt_to_equity / roe)
        operating_efficiency = np.nan if roa in [np.nan, 0] else (roe / roa)

        # Determine if company is Overvalued (1) or Undervalued (0)
        valuation = stock_valuation(pe_ratio, pb_ratio, fcf_yield)

        # Extract relevant valuation metrics
        relevant_data = {
            'Ticker': ticker,
            'P/E Ratio': pe_ratio,
            'EV/EBITDA': valuation_metrics.get('enterpriseToEbitda', np.nan),
            'Debt-to-Equity': debt_to_equity,
            'P/B Ratio': pb_ratio,
            'Revenue Growth Rate (YoY)': revenue_growth_yoy,
            'ROE': roe,
            'ROA': roa,
            'FCF Yield': fcf_yield,
            'Debt Service Ratio': debt_service_ratio,
            'Operating Efficiency': operating_efficiency,
            'Operating Efficiency': operating_efficiency,
            'Valuation':valuation
        }
        
        # Append the data to the list
        financials_list.append(relevant_data)

    except Exception as e:
        print(f"Error processing {ticker}: {e}")

# Convert the list to a DataFrame
financials = pd.DataFrame(financials_list)

# Drop duplicate tickers
financials = financials.drop_duplicates(subset=['Ticker'])

# Replace any remaining 'N/A' values with NaN
financials.replace('N/A', np.nan, inplace=True)

# Normalize specific columns in the 'financials' DataFrame


# Select columns to normalize (exclude non-numeric or categorical columns)
columns_to_normalize = ['P/E Ratio', 'EV/EBITDA', 'Debt-to-Equity', 'P/B Ratio',
                        'Revenue Growth Rate (YoY)', 'ROE', 'ROA', 'FCF Yield',
                        'Debt Service Ratio', 'Operating Efficiency']


# Create a MinMaxScaler object
scaler = MinMaxScaler()

# Convert columns to numeric before scaling
for column in columns_to_normalize:
  financials[column] = pd.to_numeric(financials[column], errors='coerce')

# Fit and transform the selected columns
financials[columns_to_normalize] = scaler.fit_transform(financials[columns_to_normalize])

# Display the normalized DataFrame
print("\nNormalized Financials DataFrame:")
print(financials)

# Save the final DataFrame to a CSV file
financials.to_csv('financials_data.csv', index=False)

columns_to_convert = ['P/E Ratio', 'EV/EBITDA', 'Debt-to-Equity', 'P/B Ratio',
                      'Revenue Growth Rate (YoY)', 'ROE', 'ROA', 'FCF Yield']

financials[columns_to_convert] = financials[columns_to_convert].apply(pd.to_numeric, errors='coerce')

financials.fillna(0, inplace=True)  # Replace NaNs with 0 (or use another strategy)

# Save the final DataFrame to a CSV file
financials.to_csv('financials_data.csv', index=False)

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import matplotlib.pyplot as plt

# Load financial dataset
financials = pd.read_csv('financials_data.csv')

# Drop 'Debt-to-Equity' and 'P/E Ratio'
financials.drop(columns=['Debt-to-Equity', 'P/E Ratio'], inplace=True, errors='ignore')

# Add Profitability Index (Proxy: ROA / P/B Ratio)
financials['Profitability Index'] = np.where(financials['P/B Ratio'] != 0, financials['ROA'] / financials['P/B Ratio'], 0) # Replaced np.inf with 0

# Select Features (X) and Target (y)
features = ['EV/EBITDA', 'P/B Ratio', 'Revenue Growth Rate (YoY)', 'ROE', 'ROA', 'FCF Yield', 'Profitability Index']
target = 'Valuation'

# Check feature correlation
correlation_matrix = financials[features].corr()
print("Feature Correlation Matrix:\n", correlation_matrix)

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(financials[features])

# Ensure all feature values are numeric
financials[features] = financials[features].apply(pd.to_numeric, errors='coerce')

# Impute NaN values with the median for each column
imputer = SimpleImputer(strategy='median')
financials[features] = imputer.fit_transform(financials[features])

# Apply PCA
pca = PCA(n_components=5)
X_pca = pca.fit_transform(X_scaled)

# Explained variance
explained_variance = pca.explained_variance_ratio_
print("Explained Variance Ratios:", explained_variance)

# Convert PCA result to DataFrame
X_pca_df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(X_pca.shape[1])])
y = financials[target]


# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_pca_df, y, test_size=0.2, random_state=42)

# Random Forest Model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Evaluation for Random Forest
print("Random Forest R² Score:", r2_score(y_test, y_pred_rf))
print("Random Forest MSE:", mean_squared_error(y_test, y_pred_rf))

# XGBoost Model
xgb_model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100, learning_rate=0.1)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

# Evaluation for XGBoost
print("XGBoost R² Score:", r2_score(y_test, y_pred_xgb))
print("XGBoost MSE:", mean_squared_error(y_test, y_pred_xgb))

# Plot Predictions
plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred_rf, label='Random Forest', alpha=0.7)
plt.scatter(y_test, y_pred_xgb, label='XGBoost', alpha=0.7)
plt.plot([0, 1], [0, 1], '--', color='gray')
plt.xlabel("Actual Valuation")
plt.ylabel("Predicted Valuation")
plt.legend()
plt.title("Model Predictions vs Actual")
plt.show()