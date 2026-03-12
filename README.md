🏠 House Price Prediction using Machine Learning
This project builds a Machine Learning model to predict house prices using the California Housing dataset. The model analyzes various housing features such as income, house age, number of rooms, population, and proximity to the ocean to estimate the median house value.
The goal is to apply data preprocessing, exploratory data analysis, and regression modeling to solve a real-world prediction problem.
📊 Project Workflow
The project follows a complete Machine Learning pipeline:
Dataset Loading
Data Cleaning & Preprocessing
Exploratory Data Analysis (EDA)
Feature Engineering
Train-Test Split
Feature Scaling
Model Training (Linear Regression)
Model Evaluation
Visualization of Results
📁 Dataset
The dataset used in this project is California Housing Dataset, which contains housing information such as:
longitude
latitude
housing_median_age
total_rooms
total_bedrooms
population
households
median_income
ocean_proximity
median_house_value (Target Variable)
🧹 Data Preprocessing
The following preprocessing steps were performed:
Handling missing values
Filling numerical missing values with mean
Dropping remaining null rows
One-hot encoding categorical feature (ocean_proximity)
Feature scaling using StandardScaler
📊 Exploratory Data Analysis
EDA was performed to understand relationships between variables:
Correlation Heatmap
Shows relationships between numerical variables.
Target Variable Distribution
Distribution of median house value.
Statistical Summary
Basic statistics of all numerical features.
🤖 Machine Learning Model
The model used in this project:
Linear Regression
Linear Regression is used to predict house prices based on multiple independent features.
📈 Model Evaluation
The model performance is evaluated using:
Mean Absolute Error (MAE)
Mean Squared Error (MSE)
Root Mean Squared Error (RMSE)
R² Score
These metrics help measure prediction accuracy.
📉 Visualization
Actual vs Predicted Prices
A scatter plot is used to compare actual house prices vs predicted house prices, helping visualize the model performance.
🛠️ Technologies Used
Python
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn
Jupyter Notebook
