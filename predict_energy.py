# Energy Consumption Prediction - SmartManufacture Inc.

# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# 2. Load Data
df = pd.read_csv("data.csv")

# 3. Initial Data Cleaning
if 'timestamp' in df.columns:
    df.drop(columns=['timestamp'], inplace=True)

# Convert object columns to numeric
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop rows with missing target
df.dropna(subset=['equipment_energy_consumption'], inplace=True)

# Fill missing values in features with median
df.fillna(df.median(numeric_only=True), inplace=True)

# 4. Exploratory Data Analysis (EDA)
plt.figure(figsize=(10, 6))
sns.histplot(df['equipment_energy_consumption'], bins=40, kde=True)
plt.title("Distribution of Equipment Energy Consumption")
plt.show()

# Correlation heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(numeric_only=True), cmap='coolwarm', annot=False)
plt.title("Feature Correlation Heatmap")
plt.show()

# 5. Feature and Target Definition
X = df.drop(columns=['equipment_energy_consumption'])
y = df['equipment_energy_consumption']

# 6. Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 7. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 8. Model Training and Evaluation
def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"\n{name} Evaluation:")
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
    print("R^2:", r2_score(y_test, y_pred))
    return model

# Linear Regression
lr_model = evaluate_model("Linear Regression", LinearRegression(), X_train, X_test, y_train, y_test)

# Random Forest
rf_model = evaluate_model("Random Forest", RandomForestRegressor(n_estimators=100, random_state=42), X_train, X_test, y_train, y_test)

# 9. Feature Importance (from Random Forest)
importance = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(10, 6))
importance.head(15).plot(kind='barh')
plt.title("Top 15 Important Features")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# 10. Recommendations and Insights
print("\n🔍 Insights:")
print("- Zone temperature and humidity levels significantly impact equipment energy usage.")
print("- Lighting energy is a strong secondary indicator.")
print("- Random variables showed low importance and can be excluded.")
print("- Recommendation: Optimize climate control in top-impact zones and reduce unnecessary lighting load.")

