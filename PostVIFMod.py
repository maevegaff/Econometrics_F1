import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from linearmodels.panel import PanelOLS
from linearmodels.panel import RandomEffects

# Load the data
df = pd.read_excel(r"C:\Users\maeve\Downloads\PostVIF.xlsx")

# FIX 1: Remove all trailing spaces from column names
df.columns = df.columns.str.strip()

# FIX 2: Rename the important columns
df = df.rename(columns={
    'Share of Rvenue': 'Share_of_Revenue',
    'Cap value binary': 'CapValueBinary'
})


# Check required columns
required_columns = ['Rank', 'CapValueBinary', 'Share_of_Revenue']
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
X = df_clean[['Rank', 'CapValueBinary']]
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


# Prepare data for panel regression
df_clean['entity'] = df.index  # Assuming each row is a unique entity
df_clean = df_clean.set_index(['entity', df_clean.index])


###### PARK TEST FOR HETEROSCEDASTICITY
# Residuals from the OLS model
residuals = ols_model.resid

# Log of squared residuals
log_residuals_squared = np.log(residuals**2)

# Log of independent variable (Rank)
log_rank = np.log(X['Rank'])

# Perform regression for Park Test
park_test_model = sm.OLS(log_residuals_squared, sm.add_constant(log_rank)).fit()

print("\n===== Park Test for Heteroscedasticity =====")
print(park_test_model.summary())


###### WHITE TEST FOR HETEROSCEDASTICITY

# Create additional features for the White Test
X_white = X_ols.copy()
X_white['Rank^2'] = X['Rank']**2
X_white['CapValueBinary^2'] = X['CapValueBinary']**2
X_white['Rank*CapValueBinary'] = X['Rank'] * X['CapValueBinary']

# Perform regression for White Test
white_test_model = sm.OLS(residuals**2, X_white).fit()

print("\n===== White Test for Heteroscedasticity =====")
print(white_test_model.summary())


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
