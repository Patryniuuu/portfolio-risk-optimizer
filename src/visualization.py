import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.ticker as mtick

def plot_efficient_frontier(returns: pd.DataFrame, user_weights: pd.Series = None, market_ticker: str = '^GSPC'):
    """
    Rysuje Granicę Efektywną, punkt GMV oraz opcjonalnie wybrany przez użytkownika portfel.
    Zwraca obiekt fig, gotowy do wyświetlenia w Streamlit.
    """
    stocks_returns = returns.drop(columns=[market_ticker], errors='ignore')
    
    # Przygotowanie parametrów
    mu = stocks_returns.mean() * 252
    Sigma = stocks_returns.cov() * 252
    num_assets = len(stocks_returns.columns)

    #Generujemy chmurę portfeli
    num_portfolios = 5000
    results = np.zeros((3, num_portfolios))

    for i in range(num_portfolios):
        w = np.random.random(num_assets)
        w /= np.sum(w)
        
        portfolio_return = np.sum(w * mu)
        portfolio_std_dev = np.sqrt(w.T @ Sigma @ w)
        
        results[0, i] = portfolio_return
        results[1, i] = portfolio_std_dev
        results[2, i] = portfolio_return / portfolio_std_dev 

    # Szukamy GMV
    def min_variance(w):
        return w.T @ Sigma @ w

    bounds = tuple((0.0, 1.0) for _ in range(num_assets))
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    init_guess = np.repeat(1/num_assets, num_assets)

    opt_gmv = minimize(min_variance, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    gmv_weights = opt_gmv.x
    gmv_return = np.sum(gmv_weights * mu)
    gmv_std_dev = np.sqrt(gmv_weights.T @ Sigma @ gmv_weights)

    # Wykres
    fig, ax = plt.subplots(figsize=(12, 8))

    # Chmura punktów
    scatter = ax.scatter(results[1,:], results[0,:], c=results[2,:], cmap='viridis', marker='o', s=10, alpha=0.3)
    fig.colorbar(scatter, ax=ax, label='Sharpe Ratio (Zysk do Ryzyka)')

    # Czerwona gwiazdka: GMV
    ax.scatter(gmv_std_dev, gmv_return, marker='*', color='red', s=400, label='Portfel GMV (Min. Ryzyko)', edgecolor='black')

    # Niebieska gwiazdka:  portfel początkowy (jeśli przekazano wagi)
    if user_weights is not None:
        #nasz model wypluwa wagi w procentach (np. 20.5), więc dzielimy przez 100
        w_user = user_weights.values / 100.0
        user_return = np.sum(w_user * mu)
        user_std_dev = np.sqrt(w_user.T @ Sigma @ w_user)
        ax.scatter(user_std_dev, user_return, marker='*', color='blue', s=400, label='Twój Wybrany Portfel', edgecolor='black')

    # Formatowanie
    ax.set_title('Markowitz: Granica Efektywna i Losowe Portfele', fontsize=16)
    ax.set_xlabel('Ryzyko Portfela (Roczne Odchylenie Standardowe)', fontsize=12)
    ax.set_ylabel('Oczekiwany Zysk Portfela (Roczne)', fontsize=12)
    ax.legend(loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.7)

    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    return fig