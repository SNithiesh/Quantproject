
"""
FULL REAL-WORLD PIPELINE USING ALL LIBRARIES
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

print("STEP 1: Downloading real stock data using yfinance")
df = yf.download("AAPL", start="2023-01-01", end="2024-01-01")
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)
df = df[["Close"]]
print(df.head())

print("\nSTEP 2: Cleaning and feature engineering (Pandas)")
df["Return"] = df["Close"].pct_change()
df.dropna(inplace=True)

print("\nSTEP 3: Numerical analysis using NumPy")
print("Average return:", np.mean(df["Return"]))
print("Volatility:", np.std(df["Return"]))

print("\nSTEP 4: Statistical test using SciPy")
stat, p = stats.shapiro(df["Return"])
print("Normality p-value:", p)

print("\nSTEP 5: Regression using statsmodels")
df["Lag1"] = df["Return"].shift(1)
df.dropna(inplace=True)
X = sm.add_constant(df["Lag1"])
y = df["Return"]
ols_model = sm.OLS(y, X).fit()
print(ols_model.summary())

print("\nSTEP 6: Machine Learning using scikit-learn")
X_ml = df[["Lag1"]]
y_ml = df["Return"]
ml_model = LinearRegression()
ml_model.fit(X_ml, y_ml)
df["ML_Prediction"] = ml_model.predict(X_ml)

print("\nSTEP 7: Volatility modeling using ARCH/GARCH")
garch = arch_model(df["Return"] * 100, vol="Garch", p=1, q=1)
garch_fit = garch.fit(disp="off")
print(garch_fit.summary())

print("\nSTEP 8: Option pricing using QuantLib")
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
flat_ts = ql.YieldTermStructureHandle(
    ql.FlatForward(0, calendar, rate, day_count)
)
flat_vol = ql.BlackVolTermStructureHandle(
    ql.BlackConstantVol(0, calendar, vol, day_count)
)

process = ql.BlackScholesProcess(spot_handle, flat_ts, flat_vol)
option.setPricingEngine(ql.AnalyticEuropeanEngine(process))

print("European Call Option Price:", option.NPV())

print("\nSTEP 9: Visualization")
plt.figure(figsize=(12,6))
plt.plot(df.index, df["Return"], label="Returns")
plt.plot(df.index, df["ML_Prediction"], label="ML Prediction")
plt.legend()
plt.title("Returns vs ML Prediction")
plt.show()

print("\nPIPELINE COMPLETE")
