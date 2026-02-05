
"""
EXAMPLE 02: RISK & DECISION-FOCUSED PIPELINE
===========================================

WHAT THIS FILE IS SUPPOSED TO DO:
--------------------------------
This example focuses on DECISION MAKING rather than prediction.

Questions answered:
1. How risky is the asset?
2. Does volatility cluster?
3. What is a fair option price?
4. How do statistics support decisions?

WHY THIS EXISTS:
----------------
In real finance jobs:
- Risk analysis > prediction
"""

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from arch import arch_model
import QuantLib as ql
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("darkgrid")

# =========================
# DATA INGESTION
# =========================
print("Downloading market data")

df = yf.download("MSFT", start="2022-01-01", end="2024-01-01")
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)
df = df[["Close"]]

# =========================
# RETURNS & RISK
# =========================
df["Return"] = df["Close"].pct_change()
df.dropna(inplace=True)

# Basic risk metrics
mean_return = np.mean(df["Return"])
std_risk = np.std(df["Return"])

print("Mean return:", mean_return)
print("Risk (std dev):", std_risk)

# =========================
# STATISTICAL CONFIDENCE
# =========================
stat, p_value = stats.shapiro(df["Return"])
print("Normality p-value:", p_value)

# =========================
# STATISTICAL MODEL
# =========================
df["Lag1"] = df["Return"].shift(1)
df.dropna(inplace=True)

X = sm.add_constant(df["Lag1"])
y = df["Return"]

ols = sm.OLS(y, X).fit()
print(ols.summary())

# =========================
# ML (SECONDARY)
# =========================
ml = LinearRegression()
ml.fit(df[["Lag1"]], df["Return"])

df["ML_Return"] = ml.predict(df[["Lag1"]])

# =========================
# VOLATILITY CLUSTERING
# =========================
garch = arch_model(df["Return"] * 100, p=1, q=1)
garch_fit = garch.fit(disp="off")
print(garch_fit.summary())

# =========================
# OPTION PRICING FOR DECISION
# =========================
ql.Settings.instance().evaluationDate = ql.Date.todaysDate()

spot = float(df["Close"].iloc[-1])
strike = spot * 1.05  # out-of-the-money
rate = 0.05
vol = np.std(df["Return"]) * np.sqrt(252)  # annualized
maturity = ql.Date(31, 12, 2025)

day_count = ql.Actual365Fixed()
calendar = ql.UnitedStates(ql.UnitedStates.NYSE)

option = ql.VanillaOption(
    ql.PlainVanillaPayoff(ql.Option.Call, strike),
    ql.EuropeanExercise(maturity)
)

process = ql.BlackScholesProcess(
    ql.QuoteHandle(ql.SimpleQuote(spot)),
    ql.YieldTermStructureHandle(
        ql.FlatForward(0, calendar, rate, day_count)
    ),
    ql.BlackVolTermStructureHandle(
        ql.BlackConstantVol(0, calendar, vol, day_count)
    )
)

option.setPricingEngine(ql.AnalyticEuropeanEngine(process))
print("Option price for decision making:", option.NPV())

# =========================
# VISUALIZATION
# =========================
plt.figure(figsize=(12,6))
sns.histplot(df["Return"], bins=40, kde=True)
plt.title("Return Distribution & Risk")
plt.show()

print("\nRISK-FOCUSED PIPELINE COMPLETED")
