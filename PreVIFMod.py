import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor


# Load the data
df = pd.read_excel(r"C:\Users\maeve\Downloads\InvertedRank .xlsx")

# FIX 1: Remove all trailing spaces from column names
df.columns = df.columns.str.strip()

# FIX 2: Rename the important columns
df = df.rename(columns={
    'Share of Rvenue': 'Share_of_Revenue',
    'Cap value binary': 'CapValueBinary'
})

# Check required columns
required_columns = ['Rank', 'CapValueBinary', 'Year', 'Share_of_Revenue']
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    raise KeyError(f"Missing columns: {missing_columns}")

# Print info
print("Column Information:")
print(df.info())
print("\nFirst few rows:")
print(df.head())

# Build clean dataframe
df_clean = df[required_columns].copy()

# Fill missing values
df_clean = df_clean.fillna(df_clean.mean())

# Regression inputs
X = df_clean[['Rank', 'CapValueBinary', 'Year']]
y = df_clean['Share_of_Revenue']


###### SCIKIT-LEARN REGRESSION

model = LinearRegression()
model.fit(X, y)

print("\n===== Scikit-Learn Regression Results =====")
print("Intercept:", model.intercept_)
print("Coefficients:", dict(zip(X.columns, model.coef_)))
print("R² Score:", model.score(X, y))


###### STATS MODELS OLS — SIGNIFICANCE TESTS

X_ols = sm.add_constant(X)
ols_model = sm.OLS(y, X_ols).fit()

print("\n\n===== OLS Regression With Statistical Significance Tests =====")
print(ols_model.summary())


# Calculate VIF for each feature
vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

print("\n===== Variance Inflation Factor (VIF) =====")
print(vif_data)

###### PLOT: ACTUAL VS PREDICTED + BEST-FIT LINE

y_pred = model.predict(X)

plt.figure(figsize=(8,6))

# Scatter plot of actual vs predicted
plt.scatter(y, y_pred, alpha=0.7, label="Actual vs Predicted")

# Best-fit line
z = np.polyfit(y, y_pred, 1)           # slope & intercept
p = np.poly1d(z)
plt.plot(y, p(y), color='red', label="Line of Best Fit")

plt.xlabel("Actual Share of Revenue")
plt.ylabel("Predicted Share of Revenue")
plt.title("Multiple Regression: Actual vs Predicted Share of Revenue")
plt.legend()
plt.grid(True)
plt.show()
