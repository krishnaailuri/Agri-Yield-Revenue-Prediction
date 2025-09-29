import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# 1. Load Data
df = pd.read_csv("./data/corn_yield.csv")

print("Initial shape:", df.shape)
print(df.info())
print("\nMissing values before cleaning:")
print(df.isnull().sum())

# 2. Drop useless/null columns
cols_to_drop = [
    "Week Ending", "Ag District", "Ag District Code", "County", "County ANSI",
    "Zip Code", "Region", "Watershed", "CV (%)", "watershed_code"
]
df.drop(columns=cols_to_drop, inplace=True, errors='ignore')

# 3. Fix missing 'State ANSI'
if "State ANSI" in df.columns:
    df["State ANSI"] = df["State ANSI"].fillna(0).astype(int)

# 4. Clean target 'Value'
df["Value"] = df["Value"].astype(str).str.replace(",", "", regex=False)
df["Value"] = pd.to_numeric(df["Value"], errors="coerce")

# 5. Keep only relevant columns
keep_cols = ["Year", "Period", "State", "Data Item", "Domain",
             "Domain Category", "Commodity", "State ANSI", "Value"]
df = df[[c for c in keep_cols if c in df.columns]].copy()

# 6. Prepare features and target
X = df.drop("Value", axis=1)
y = df["Value"]

# Remove rows with NaN in target
mask = y.notnull()
X = X.loc[mask, :]
y = y.loc[mask]

# 7. Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=27)

# 8. Define preprocessing and pipeline
categorical_cols = ["Period", "State", "Data Item", "Domain", "Domain Category", "Commodity"]
numeric_cols = ["Year", "State ANSI"]

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols)
], remainder="passthrough")

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(n_estimators=100, random_state=0))
])

# 9. Train model
pipeline.fit(X_train, y_train)

# 10. Predict and evaluate
y_pred = pipeline.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nMean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")
print(f"R2 Score: {r2}")

# 11. Feature importances (after encoding)
# Extract feature names after one-hot encoding
encoded_feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
feat_imp = pd.Series(pipeline.named_steps['model'].feature_importances_, index=encoded_feature_names)
feat_imp = feat_imp.sort_values(ascending=False)
print("\nTop 20 feature importances:")
print(feat_imp.head(20))

# 12. Plot predicted vs actual
plt.figure(figsize=(8,6))
sns.regplot(x=y_pred, y=y_test)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Predicted vs Actual Values")
plt.show()

# 13. Save pipeline for Streamlit UI
with open("crop_price_pipeline.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("\nTrained pipeline saved as 'crop_price_pipeline.pkl'")
