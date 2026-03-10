# 📈 Quantitative Portfolio Analyzer & Optimizer

An interactive web application built with Python and Streamlit that allows users to perform advanced quantitative analysis and optimize their stock portfolios based on Modern Portfolio Theory (MPT).

🔗 **[Live Demo: Try the App Here]** *(Replace this with your Streamlit Cloud link once deployed)*

## 🧠 Key Features

This application implements several core concepts from quantitative finance:
* **Capital Asset Pricing Model (CAPM):** Calculates Alpha, Beta, Market Risk ($R^2$), and Specific Risk using linear regression.
* **Tail Risk Analysis:** Computes Value at Risk (VaR) and Conditional Value at Risk (CVaR) at a 95% confidence level using historical simulation.
* **Markowitz Portfolio Optimization (MPT):** Uses sequential least squares programming (`SciPy SLSQP`) to find the **Global Minimum Variance (GMV)** portfolio and the optimal weights for any user-defined target return.
* **Efficient Frontier Visualization:** Generates a Monte Carlo simulation (5,000 random portfolios) and plots the Efficient Frontier with dynamic markers for the GMV and user-selected portfolios.

## 🛠️ Tech Stack
* **Frontend:** Streamlit
* **Data Ingestion:** yfinance (Yahoo Finance API)
* **Mathematical Engine:** NumPy, SciPy (`scipy.optimize`, `scipy.stats`), Pandas
* **Visualization:** Matplotlib

## 💻 How to Run Locally

If you want to run this application on your own machine, follow these steps:

1. Clone the repository:
   ```bash
   git clone [https://github.com/TwojNick/portfolio-optimizer-app.git](https://github.com/TwojNick/portfolio-optimizer-app.git)
   cd portfolio-optimizer-app
   ```
2.Install the required dependencies:
  ```bash
  pip install -r requirements.txt
  ```
3.Run the Streamlit app:
  ```bash
  streamlit run app.py
  ```

💡 Note: The mathematical derivations and exploratory data analysis (EDA) process are documented in Polish within the included Jupyter Notebook (01_theory_and_logic_walkthrough.ipynb), while this main application is designed for a global audience.