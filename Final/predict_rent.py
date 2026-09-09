import numpy as np
import pandas as pd

feature_cols = [
    'BEDS',
    'BATHS',
    'SQFT',
    'BUILDING_TYPE_APT',
    'BUILDING_TYPE_COMM',
    'BUILDING_TYPE_CON',
    'BUILDING_TYPE_SFR',
    'GARAGE_Y',
    'POOL_Y',
    'TIME'
]

# Load saved ZIP-level regression parameters
zip_coefficients = pd.read_parquet(
    "zip_regression_coefficients.parquet"
)

# Make sure ZIP has the SAME type in both dataframes.
new_data = new_data.copy()
new_data[feature_cols] = new_data[feature_cols].fillna(0)

# Merge the relevant ZIP's coefficients onto every listing
estimated_data = new_data.merge(
    zip_coefficients,
    on='ZIP',
    how='left',
    validate='m:1'   # many listings may map to one ZIP coefficient vector
)

# Initialize with the ZIP-specific intercept
estimated_data['predicted_logrent'] = estimated_data['intercept']

# Add X beta, using the saved coefficient for each feature
for feature in feature_cols:
    estimated_data['predicted_logrent'] += (
        estimated_data[feature] * estimated_data[f'coef_{feature}']
    )

# Convert back to rent levels if logrent was constructed with np.log(rent)
estimated_data['predicted_rent'] = np.exp(
    estimated_data['predicted_logrent']
)
