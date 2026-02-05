
"""
EXAMPLE 03: PORTFOLIO-LEVEL REAL-WORLD ANALYSIS
=============================================

WHAT THIS SCRIPT IS SUPPOSED TO DO:
----------------------------------
This script demonstrates how professionals analyze a PORTFOLIO
(not just a single stock).

In real finance roles, money is almost never invested in one asset.
Portfolio-level thinking is mandatory.

WHAT QUESTIONS WE ANSWER:
-------------------------
1. What is the portfolio return?
2. How risky is the portfolio?
3. Do portfolio returns look statistically normal?
4. Does volatility cluster over time?
5. How can this information support decisions?

WHY THIS SCRIPT USES MANY LIBRARIES:
-----------------------------------
Each library solves ONE specific problem well.
Real projects combine many libraries together.
"""

# ===============================
# IMPORTS
# ===============================

import numpy as np              # Numerical computations
import pandas as pd             # Tabular data handling
import yfinance as yf           # Market data source

from scipy import stats         # Statistical tests
import statsmodels.api as sm    # Statistical models

from arch import arch_model     # Volatility models (GARCH)
import QuantLib as ql           # Financial pricing library

import matplotlib.pyplot as plt # Core plotting
import seaborn as sns           # Better-looking plots

sns.set_style("darkgrid")

# ===============================
# STEP 1: DOWNLOAD MARKET DATA
# ===============================

# We choose multiple large-cap stocks
assets = ["AAPL", "MSFT", "GOOGL"]

print("Downloading price data for assets:", assets)

# yfinance returns a Pandas DataFrame automatically
df_download = yf.download(assets, start="2022-01-01", end="2024-01-01")

# Handle MultiIndex columns (Attribute, Ticker) or (Price, Attribute, Ticker)
if isinstance(df_download.columns, pd.MultiIndex):
    if "Close" in df_download.columns.get_level_values(0):
        prices = df_download["Close"]
    elif "Close" in df_download.columns.get_level_values(1):
        # Handle case where Attribute might be at level 1
        prices = df_download.xs("Close", level=1, axis=1)
    else:
        # Last resort: just use the DataFrame as is
        prices = df_download
else:
    prices = df_download[["Close"]]

print("\nPrice data (head):")
print(prices.head())

# ===============================
# STEP 2: CALCULATE RETURNS
# ===============================

# Finance analysis is done on RETURNS, not prices
returns = prices.pct_change().dropna()

print("\nDaily returns (head):")
print(returns.head())

# ===============================
# STEP 3: BUILD A PORTFOLIO
# ===============================

# Portfolio weights must sum to 1
weights = np.array([0.4, 0.3, 0.3])

# Matrix multiplication: (returns × weights)
portfolio_returns = returns @ weights

print("\nPortfolio returns (head):")
print(portfolio_returns.head())

# ===============================
# STEP 4: NUMERICAL RISK METRICS
# ===============================

mean_return = np.mean(portfolio_returns)
risk = np.std(portfolio_returns)

print("\nPortfolio mean return:", mean_return)
print("Portfolio risk (volatility):", risk)

# WHY THIS MATTERS:
# Mean → performance
# Volatility → uncertainty / risk

# ===============================
# STEP 5: STATISTICAL VALIDATION
# ===============================

# Check if portfolio returns are normally distributed
stat, p_value = stats.shapiro(portfolio_returns)

print("\nNormality test p-value:", p_value)

# Interpretation:
# p < 0.05 → returns are NOT normal (common in markets)

# ===============================
# STEP 6: MARKET SENSITIVITY MODEL
# ===============================

# Use average market return as a simple proxy
market_return = returns.mean(axis=1)

X = sm.add_constant(market_return)
y = portfolio_returns

regression = sm.OLS(y, X).fit()

print("\nPortfolio vs Market Regression:")
print(regression.summary())

# WHY THIS MATTERS:
# Helps understand sensitivity to broad market moves

# ===============================
# STEP 7: VOLATILITY MODELING (GARCH)
# ===============================

# GARCH models volatility clustering
garch = arch_model(portfolio_returns * 100, p=1, q=1)

garch_fit = garch.fit(disp="off")

print("\nGARCH Model Summary:")
print(garch_fit.summary())

# ===============================
# STEP 8: DECISION SUPPORT (OPTION PRICING)
# ===============================

# We use portfolio risk as a proxy for volatility
ql.Settings.instance().evaluationDate = ql.Date.todaysDate()

spot_price = 100.0
strike_price = 105.0
risk_free_rate = 0.05
volatility = risk * np.sqrt(252)  # annualized volatility
maturity = ql.Date(31, 12, 2025)

day_count = ql.Actual365Fixed()
calendar = ql.UnitedStates(ql.UnitedStates.NYSE)

option = ql.VanillaOption(
    ql.PlainVanillaPayoff(ql.Option.Call, strike_price),
    ql.EuropeanExercise(maturity)
)

process = ql.BlackScholesProcess(
    ql.QuoteHandle(ql.SimpleQuote(spot_price)),
    ql.YieldTermStructureHandle(
        ql.FlatForward(0, calendar, risk_free_rate, day_count)
    ),
    ql.BlackVolTermStructureHandle(
        ql.BlackConstantVol(0, calendar, volatility, day_count)
    )
)

option.setPricingEngine(ql.AnalyticEuropeanEngine(process))

print("\nOption price (decision support):", option.NPV())

# ===============================
# STEP 9: VISUALIZATION
# ===============================

plt.figure(figsize=(12, 6))
sns.histplot(portfolio_returns, bins=40, kde=True)
plt.title("Portfolio Return Distribution")
plt.xlabel("Daily Return")
plt.ylabel("Frequency")
plt.show()

print("\nPORTFOLIO ANALYSIS COMPLETE")
