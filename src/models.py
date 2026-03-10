import pandas as pd
import numpy as np
from scipy.optimize import minimize
import streamlit as st
from scipy import stats

@st.cache_data(show_spinner=False)
def calculate_capm(returns: pd.DataFrame, market_ticker: str = '^GSPC') -> pd.DataFrame:
    """
    Wylicza parametry modelu CAPM (Alpha, Beta, R-squared) dla każdej spółki.
    Używa scipy.stats.linregress.
    """
    results = {}
    market_returns = returns[market_ticker]
    stocks_returns = returns.drop(columns=[market_ticker], errors='ignore')
    
    for column in stocks_returns.columns:
        # Odpalamy pełną regresję liniową: X = rynek, Y = spółka
        lin_reg = stats.linregress(market_returns, stocks_returns[column])
        
        # Wyciągamy statystyki z modelu
        beta = lin_reg.slope            # Współczynnik kierunkowy
        alpha = lin_reg.intercept       # Wyraz wolny (przecięcie z osią Y)
        r_squared = lin_reg.rvalue ** 2 # Współczynnik determinacji (R^2)
        
        results[column] = {
            'Beta CAPM': round(beta, 2),
            'Alpha (Roczna %)': round(alpha * 252 * 100, 2),
            'Ryzyko Rynkowe R^2 (%)': round(r_squared * 100, 2),
            'Ryzyko Specyficzne (%)': round((1 - r_squared) * 100, 2)
        }
        
    return pd.DataFrame(results).T

@st.cache_data(show_spinner=False)
def calculate_var_cvar(returns: pd.DataFrame, alpha: float = 0.05, market_ticker: str = '^GSPC') -> pd.DataFrame:
    """
    Wylicza Value at Risk (VaR) i Conditional Value at Risk (CVaR).
    """
    results = {}
    
    # Pracujemy tylko na akcjach, wyrzucamy benchmark
    assets_returns = returns.drop(columns=[market_ticker], errors='ignore')
    
    for column in assets_returns.columns:
        stock_rets = assets_returns[column]
        
        # VaR: Kwantyl alpha%
        var = stock_rets.quantile(alpha)
        # CVaR: Średnia ze strat gorszych niż VaR
        cvar = stock_rets[stock_rets <= var].mean()
        
        results[column] = {
            f'VaR ({(1-alpha)*100}%) w %': round(abs(var) * 100, 2),
            f'CVaR ({(1-alpha)*100}%) w %': round(abs(cvar) * 100, 2)
        }
        
        
    return pd.DataFrame(results).T

@st.cache_data(show_spinner=False)
def optimize_portfolio(returns: pd.DataFrame, target_return: float = None, market_ticker: str = '^GSPC') -> pd.Series:
    """
    Silnik optymalizacyjny Markowitza.
    Jeśli target_return to None -> wylicza Global Minimum Variance (GMV).
    Jeśli target_return jest podany -> wylicza optymalny portfel Mean-Variance dla zadanego zysku.
    """
    # Wyrzucamy benchmark do optymalizacji
    assets_returns = returns.drop(columns=[market_ticker], errors='ignore')
    
    mu = assets_returns.mean() * 252
    Sigma = assets_returns.cov() * 252
    num_assets = len(assets_returns.columns)
    
    # Funkcja celu: Wariancja
    def portfolio_variance(weights):
        return weights.T @ Sigma @ weights
        
    init_guess = np.repeat(1 / num_assets, num_assets)
    bounds = tuple((0.0, 1.0) for _ in range(num_assets))
    
    # Podstawowe ograniczenie: wagi sumują się do 1
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    
    # Ograniczenie dla zadanego zysku
    if target_return is not None:
        constraints.append({'type': 'eq', 'fun': lambda x: np.sum(x * mu) - target_return})
        
    # Odpalamy solver
    result = minimize(
        fun=portfolio_variance,
        x0=init_guess,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    # Zwracamy wagi w procentach
    return pd.Series(np.round(result.x * 100, 2), index=assets_returns.columns, name="Wagi (%)")