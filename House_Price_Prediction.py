import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

"""

2. Load Dataset"""

import pandas as pd
df=pd.read_csv("housing.csv")

df.head()

"""3. Data Cleaning & Preprocessing"""

df.isnull().sum()

""" Handle Missing Values"""

df.fillna(df.mean(numeric_only=True), inplace=True)
df.dropna(inplace=True)

"""🔹 4. Exploratory Data Analysis"""

df.info()

"""Statistical Summary"""

df.describe()

"""Correlation Heatmap"""

plt.figure(figsize=(10,6))
sns.heatmap(df.select_dtypes(include=np.number).corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

"""Target Variable Distribution"""

sns.histplot(df['median_house_value'], kde=True)
plt.title("Median House Value Distribution")
plt.show()

"""5.Feature Selection"""

X = df.drop('median_house_value', axis=1)

# Handle categorical features using one-hot encoding
X = pd.get_dummies(X, columns=['ocean_proximity'], drop_first=True)

y = df['median_house_value']

"""6. Train-Test Split"""

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

"""7. Feature Scaling"""

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

"""8. Linear Regression Model Building"""

model = LinearRegression()
model.fit(X_train_scaled, y_train)

"""9. Model Prediction"""

y_pred = model.predict(X_test_scaled)

"""10. Model Evaluation"""

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("R² Score:", r2)

"""11. Actual vs Predicted Visualization"""

plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         'r--')

plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")
plt.show()

r2 = r2_score(y_test, y_pred)
print(f"R² Score: {r2:.4f}")
