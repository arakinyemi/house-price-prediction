import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
import pickle
import os

# 1. Generate Synthetic Data
print("Generating synthetic data...")
np.random.seed(42)
n_samples = 1000

data = {
    'OverallQual': np.random.randint(1, 11, n_samples),
    'GrLivArea': np.random.randint(500, 4000, n_samples),
    'TotalBsmtSF': np.random.randint(0, 3000, n_samples),
    'GarageCars': np.random.randint(0, 5, n_samples),
    'YearBuilt': np.random.randint(1950, 2023, n_samples),
    'FullBath': np.random.randint(1, 4, n_samples)
}

df = pd.DataFrame(data)

# Generate a synthetic SalePrice
df['SalePrice'] = (
    df['OverallQual'] * 20000 +
    df['GrLivArea'] * 100 +
    df['TotalBsmtSF'] * 50 +
    df['GarageCars'] * 15000 +
    (df['YearBuilt'] - 1950) * 500 +
    df['FullBath'] * 10000 +
    np.random.normal(0, 20000, n_samples)
)

# 2. Preprocessing
features = ['OverallQual', 'GrLivArea', 'TotalBsmtSF', 'GarageCars', 'YearBuilt', 'FullBath']
target = 'SalePrice'

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Model Pipeline
# Matching the user's notebook pipeline structure
model_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# 4. Train
print("Training model...")
model_pipeline.fit(X_train, y_train)

# 5. Save Model
output_dir = 'model'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

model_path = os.path.join(output_dir, 'house_price_model.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(model_pipeline, f)
print(f"Model saved to {model_path}")
