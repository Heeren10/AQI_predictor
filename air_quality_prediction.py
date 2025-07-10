import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# -------------------------------
# Load Dataset
# -------------------------------
df = pd.read_csv('dataset/air_quality_data.csv')

# -------------------------------
# Step 1: Feature Engineering
# -------------------------------
df['PM_ratio'] = df['PM2.5'] / (df['PM10'] + 1e-3)  # Avoid division by zero
df['date'] = pd.date_range(start='2020-01-01', periods=len(df), freq='D')
df['day_of_week'] = df['date'].dt.dayofweek
df['month'] = df['date'].dt.month
df.drop(columns=['date'], inplace=True)

# -------------------------------
# Define Features and Target
# -------------------------------
features = [
    'PM2.5', 'PM10', 'NO2', 'CO', 'temperature', 'humidity',
    'wind_speed', 'traffic_density', 'PM_ratio', 'day_of_week', 'month'
]
target = 'AQI_next_24h'

X = df[features]
y = df[target]

# -------------------------------
# Step 2: Outlier Removal
# -------------------------------
iso = IsolationForest(contamination=0.01, random_state=42)
outliers = iso.fit_predict(X)
X = X[outliers == 1]
y = y[outliers == 1]

# -------------------------------
# Step 3: Train/Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# Train Model
# -------------------------------
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Predict and Evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n✅ Model trained successfully!")
print(f"📊 Mean Squared Error: {mse:.2f}")
print(f"📈 R² Score: {r2:.4f}")

# -------------------------------
# Save Model
# -------------------------------
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/trained_model_enhanced.pkl")
print("💾 Model saved to: models/trained_model_enhanced.pkl")

# -------------------------------
# Plot Feature Importance
# -------------------------------
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
sorted_features = np.array(features)[indices]

plt.figure(figsize=(10, 6))
sns.barplot(x=sorted_features, y=importances[indices])
plt.title("🔍 Feature Importance (Enhanced Model)")
plt.xticks(rotation=45)
plt.tight_layout()

# Save the plot
os.makedirs("visualizations", exist_ok=True)
plt.savefig("visualizations/feature_importance_enhanced.png")
print("📊 Feature importance saved to: visualizations/feature_importance_enhanced.png")
plt.show()
