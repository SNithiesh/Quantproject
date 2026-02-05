"""
NIFTY 50: RISK & DECISION-FOCUSED PIPELINE
=========================================

WHAT THIS FILE IS SUPPOSED TO DO:
--------------------------------
This example focuses on DECISION MAKING rather than prediction,
applied specifically to the NIFTY 50 Index (^NSEI).

Questions answered:
1. How risky is the NIFTY 50 index?
2. Does volatility cluster in the Indian market?
3. What is a fair option price for the index?
4. How do statistics support decisions?

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
print("Downloading NIFTY 50 market data")

# Using ticker "^NSEI" for Nifty 50
# Adjusted dates to be relevant for the current time (2026)
df = yf.download("^NSEI", start="2024-01-01", end="2026-01-01")

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

print(f"Mean return: {mean_return:.6f}")
print(f"Risk (std dev): {std_risk:.6f}")

# =========================
# STATISTICAL CONFIDENCE
# =========================
stat, p_value = stats.shapiro(df["Return"])
print(f"Normality p-value: {p_value:.6f}")

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
# Modeling volatility for Indian Markets
garch = arch_model(df["Return"] * 100, p=1, q=1)
garch_fit = garch.fit(disp="off")
print(garch_fit.summary())

# =========================
# OPTION PRICING FOR DECISION
# =========================
ql.Settings.instance().evaluationDate = ql.Date.todaysDate()

spot = float(df["Close"].iloc[-1])
strike = spot * 1.05  # out-of-the-money
rate = 0.065  # Approximate risk-free rate for India (higher than US)
vol = np.std(df["Return"]) * np.sqrt(252)  # annualized
maturity = ql.Date(31, 12, 2027)  # Adjusted for future maturity

day_count = ql.Actual365Fixed()
# Using India calendar if available, otherwise fallback (QuantLib supports India)
calendar = ql.India()

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
plt.title("NIFTY 50 Return Distribution & Risk")
plt.show()

print("\nNIFTY 50 RISK PIPELINE COMPLETED")
