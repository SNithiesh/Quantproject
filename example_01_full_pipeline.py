
"""
EXAMPLE 01: FULL FINANCIAL ANALYSIS PIPELINE (EDUCATIONAL)
========================================================

WHAT THIS FILE IS SUPPOSED TO DO:
--------------------------------
This file demonstrates a COMPLETE real-world workflow using ALL libraries together.

We act as a data/finance analyst working on a stock.

Pipeline:
1. Get real market data (yfinance)
2. Clean and prepare data (Pandas)
3. Perform numerical analysis (NumPy)
4. Run statistical tests (SciPy)
5. Do regression analysis (statsmodels)
6. Build a predictive model (scikit-learn)
7. Model volatility (ARCH/GARCH)
8. Price an option (QuantLib)
9. Visualize results (matplotlib + seaborn)

WHY THIS EXISTS:
----------------
This mirrors how analytics & quant teams work end-to-end.
"""

# =========================
# IMPORTS
# =========================
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
# STEP 1: GET REAL DATA
# =========================
print("Downloading real stock data (AAPL)")

df = yf.download("AAPL", start="2023-01-01", end="2024-01-01")
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# WHY:
# yfinance gives us a Pandas DataFrame directly
# Pandas is the STANDARD data container

df = df[["Close"]]
print(df.head())

# =========================
# STEP 2: DATA PREPARATION
# =========================
print("\nCalculating returns using Pandas")

df["Return"] = df["Close"].pct_change()
df.dropna(inplace=True)

# WHY RETURNS:
# Finance models behavior on percentage change, not raw price

# =========================
# STEP 3: NUMERICAL ANALYSIS
# =========================
print("\nNumerical analysis (NumPy)")

avg_return = np.mean(df["Return"])
volatility = np.std(df["Return"])

print("Average daily return:", avg_return)
print("Volatility (risk):", volatility)

# =========================
# STEP 4: STATISTICAL TESTS
# =========================
print("\nStatistical test (SciPy)")

stat, p_value = stats.shapiro(df["Return"])

print("Normality test p-value:", p_value)

# INTERPRETATION:
# p < 0.05 → returns are NOT normally distributed (very common)

# =========================
# STEP 5: REGRESSION (STATSMODELS)
# =========================
print("\nRegression analysis (statsmodels)")

df["Lag1"] = df["Return"].shift(1)
df.dropna(inplace=True)

X = sm.add_constant(df["Lag1"])
y = df["Return"]

ols_model = sm.OLS(y, X).fit()
print(ols_model.summary())

# WHY STATSMODELS:
# We want coefficients, p-values, and inference

# =========================
# STEP 6: MACHINE LEARNING
# =========================
print("\nMachine learning prediction (scikit-learn)")

X_ml = df[["Lag1"]]
y_ml = df["Return"]

ml_model = LinearRegression()
ml_model.fit(X_ml, y_ml)

df["ML_Prediction"] = ml_model.predict(X_ml)

# WHY SCIKIT-LEARN:
# Focused on prediction accuracy, not statistical explanation

# =========================
# STEP 7: VOLATILITY MODELING
# =========================
print("\nVolatility modeling (ARCH/GARCH)")

returns_pct = df["Return"] * 100  # ARCH expects %

garch = arch_model(returns_pct, vol="Garch", p=1, q=1)
garch_fit = garch.fit(disp="off")

print(garch_fit.summary())

# =========================
# STEP 8: OPTION PRICING
# =========================
print("\nOption pricing (QuantLib)")

ql.Settings.instance().evaluationDate = ql.Date.todaysDate()

spot = float(df["Close"].iloc[-1])
strike = spot
rate = 0.05
vol = 0.25
maturity = ql.Date(31, 12, 2026)

day_count = ql.Actual365Fixed()
calendar = ql.UnitedStates(ql.UnitedStates.NYSE)

payoff = ql.PlainVanillaPayoff(ql.Option.Call, strike)
exercise = ql.EuropeanExercise(maturity)
option = ql.VanillaOption(payoff, exercise)

spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot))
yield_curve = ql.YieldTermStructureHandle(
    ql.FlatForward(0, calendar, rate, day_count)
)
vol_surface = ql.BlackVolTermStructureHandle(
    ql.BlackConstantVol(0, calendar, vol, day_count)
)

process = ql.BlackScholesProcess(
    spot_handle, yield_curve, vol_surface
)

option.setPricingEngine(ql.AnalyticEuropeanEngine(process))

print("European Call Option Price:", option.NPV())

# =========================
# STEP 9: VISUALIZATION
# =========================
print("\nVisualizing results")

plt.figure(figsize=(12,6))
plt.plot(df.index, df["Return"], label="Actual Returns")
plt.plot(df.index, df["ML_Prediction"], label="ML Prediction")
plt.legend()
plt.title("Returns vs ML Prediction")
plt.show()

print("\nPIPELINE COMPLETED")
