# Yahoo Finance data fetcher
import yfinance as yf

class DataFetcher:
    def __init__(self, ticker):
        self.ticker = ticker

    def fetch_data(self):
        return yf.download(self.ticker)
