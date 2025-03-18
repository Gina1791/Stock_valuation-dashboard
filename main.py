
import yfinance as yf

tickers = ["AAPL", "MSFT", "GOOGL","NVDA","BRK.B","AMZN","META"]  # List of tickers
start_date = "2022-01-01"
end_date = "2025-02-18"

# Download historical data
data = yf.download(ticker, start=start_date, end=end_date)
print(data.head())  # Display the first few rows of the data
