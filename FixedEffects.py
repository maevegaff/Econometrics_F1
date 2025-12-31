import pandas as pd
from linearmodels.panel import PanelOLS
from statsmodels.tools.tools import add_constant


# Load the Excel file
df = pd.read_excel(r"C:\Users\maeve\Downloads\FixedEffectsSpreadSheet.xlsx")

# Set the panel index (entity, time)
df = df.set_index(['Team', 'Year'])

# Define dependent and independent variables
y = df['Share of Rvenue']
X = df[['Rank', 'Cap value binary']]

# Optional: add constant (FE will absorb it, but allowed)
X = add_constant(X)

# Estimate Fixed Effects model (entity FE)
fe_model = PanelOLS(
    y,
    X,
    entity_effects=True
)

# Fit with robust (clustered) standard errors
fe_results = fe_model.fit(
    cov_type='clustered',
    cluster_entity=True
)

# Print results
print(fe_results.summary)
