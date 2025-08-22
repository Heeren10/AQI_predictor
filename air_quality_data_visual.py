# Air Quality Dataset Visualization & Analysis
# Target: AQI_next_24h

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# 1. Load Dataset
data = pd.read_csv("air_quality_data.csv")

print("Dataset Info:")
print(data.info())
print("\nFirst 5 rows:")
print(data.head())


# 2. Summary Statistics
print("\nSummary Statistics:")
print(data.describe())


# 3. Distribution of Target (AQI_next_24h)
plt.figure(figsize=(8,5))
sns.histplot(data['AQI_next_24h'], bins=30, kde=True, color='purple')
plt.title("Distribution of AQI (Next 24 Hours)")
plt.xlabel("AQI_next_24h")
plt.ylabel("Count")
plt.show()


# 4. Distribution of Features
data.hist(bins=30, figsize=(12,8), color='skyblue', edgecolor='black')
plt.suptitle("Feature Distributions")
plt.show()


# 5. Correlation Heatmap
plt.figure(figsize=(10,6))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()


# 6. Scatterplots (Feature vs Target)
features = ['PM2.5','PM10','NO2','CO','temperature','humidity','wind_speed','traffic_density']
for col in features:
    plt.figure(figsize=(6,4))
    sns.scatterplot(x=data[col], y=data['AQI_next_24h'], alpha=0.6)
    plt.title(f"{col} vs AQI_next_24h")
    plt.xlabel(col)
    plt.ylabel("AQI_next_24h")
    plt.show()


# 7. Pairplot (Multiple Features vs Target)
sns.pairplot(data, x_vars=features, y_vars='AQI_next_24h', height=3, aspect=0.8, kind='scatter')
plt.show()


# 8. Boxplots (Outlier Detection for All Features)
plt.figure(figsize=(12,6))
sns.boxplot(data=data[features + ['AQI_next_24h']], orient="h")
plt.title("Outlier Detection in Features & Target")
plt.show()


# 9. Relationship Between Pollution Features
plt.figure(figsize=(10,6))
sns.scatterplot(x=data['PM2.5'], y=data['PM10'], hue=data['AQI_next_24h'], palette="viridis", alpha=0.7)
plt.title("PM2.5 vs PM10 colored by AQI_next_24h")
plt.show()
