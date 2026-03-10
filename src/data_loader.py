import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st

@st.cache_data(show_spinner=False)
def fetch_stock_data(tickers: list, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Pobiera historyczne dzienne ceny zamknięcia (Close) z Yahoo Finance.
    Wymagany format daty: 'YYYY-MM-DD'.
    """
    # Zawsze doklejamy indeks S&P 500 (^GSPC), bo jest nam niezbędny do modelu CAPM
    all_tickers = tickers + ['^GSPC']
    
    data = yf.download(all_tickers, start=start_date, end=end_date, progress=False)
    
    # Zabezpieczenie na wypadek zmiany struktury pobieranych danych przez yfinance
    if isinstance(data.columns, pd.MultiIndex):
        prices = data['Close']
    else:
        prices = data
        
    prices = prices.dropna()
    
    return prices

@st.cache_data(show_spinner=False)
def calculate_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Konwertuje surowe ceny na dzienne stopy zwrotu logarytmicznego.
    Matematyka: ln(Cena_dzisiaj / Cena_wczoraj)
    """
    # Wyliczamy log-zwroty za pomocą numpy
    log_returns = np.log(prices / prices.shift(1))
    
    # Pierwszy wiersz zawsze staje się NaN (nie ma wczorajszej ceny), więc go ucinamy
    log_returns = log_returns.dropna()
    
    return log_returns