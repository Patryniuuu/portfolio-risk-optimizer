import streamlit as st
import pandas as pd
import datetime

from src import data_loader as dl
from src import models as md
from src import visualization as vis

st.set_page_config(page_title="Portfolio Analyzer", layout="wide")

st.title("📈 Zaawansowana Analiza Portfela Inwestycyjnego")
st.markdown("Aplikacja oparta na modelu CAPM, analizie ryzyka skrajnego (VaR/CVaR) oraz optymalizacji Markowitza, dla spółek rynku amerykańskiego.")

# PANEL BOCZNY (SIDEBAR)
st.sidebar.header("Parametry Analizy")

tickers_input = st.sidebar.text_input(
    "Podaj tickery spółek (oddzielone przecinkiem):", 
    value="AAPL, MSFT, NVDA, KO"
)

start_date = st.sidebar.date_input("Data początkowa", datetime.date(2021, 1, 1))
end_date = st.sidebar.date_input("Data końcowa", datetime.date(2026, 3, 1))

# Przycisk uruchamiający całą machinę
run_button = st.sidebar.button("Uruchom Analizę", type="primary")

if "analiza_uruchomiona" not in st.session_state:
    st.session_state.analiza_uruchomiona = False

if run_button:
    st.session_state.analiza_uruchomiona = True

if st.session_state.analiza_uruchomiona:
    tickers_list = [ticker.strip() for ticker in tickers_input.split(',')]
    if "^GSPC" not in tickers_list:
        tickers_list.append("^GSPC")
    
    with st.spinner("Pobieranie danych z Yahoo Finance..."):
        prices = dl.fetch_stock_data(tickers_list, start_date, end_date)
        returns = dl.calculate_log_returns(prices)
        
    st.success("Dane pobrane i przetworzone pomyślnie!")
    
    tab_capm, tab_risk, tab_markowitz = st.tabs(["1. Model CAPM", "2. Ryzyko VaR / CVaR", "3. Optymalizacja Markowitza"])
    
    with tab_capm:
        st.subheader("Wycena Aktywów - Model CAPM")
        capm_results = md.calculate_capm(returns)
        st.dataframe(capm_results, use_container_width=True)
        
    with tab_risk:
        st.subheader("Ryzyko Skrajne (Grube Ogony)")
        risk_results = md.calculate_var_cvar(returns)
        st.dataframe(risk_results, use_container_width=True)
        
    with tab_markowitz:
        st.subheader("Global Minimum Variance & Efektywna Granica")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### Twój cel inwestycyjny")
            target_return_slider = st.slider(
                "Oczekiwany roczny zysk", 
                min_value=0.0, max_value=1.0, value=0.25, step=0.01, format="%.2f"
            )
            
            st.markdown("#### Optymalne wagi portfela:")
            user_weights = md.optimize_portfolio(returns, target_return=target_return_slider)
            st.dataframe(user_weights, use_container_width=True)
            
        with col2:
            st.markdown("### Granica Efektywna")
            with st.spinner("Generowanie symulacji Monte Carlo..."):
                fig = vis.plot_efficient_frontier(returns, user_weights)
                st.pyplot(fig)
                
else:
    st.info("👈 Skonfiguruj parametry w panelu bocznym i kliknij 'Uruchom Analizę'.")

