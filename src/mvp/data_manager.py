import yfinance as yf
import os
import pickle

class DataManager:
    def __init__(self, cache_file='data_cache.pkl'):
        self.cache_file = cache_file
        self.data_cache = self.load_cache()

    def load_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'rb') as f:
                return pickle.load(f)
        return {}

    def fetch_data(self, ticker):
        if ticker in self.data_cache:
            return self.data_cache[ticker]
        else:
            data = yf.download(ticker)
            self.data_cache[ticker] = data
            self.save_cache()
            return data

    def save_cache(self):
        with open(self.cache_file, 'wb') as f:
            pickle.dump(self.data_cache, f)
