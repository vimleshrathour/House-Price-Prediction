🏠 House Price Prediction using Machine Learning

This project builds a Machine Learning model to predict house prices using the California Housing Dataset. The model analyzes housing features such as income, house age, number of rooms, population, and proximity to the ocean to estimate the median house value.

The main objective of this project is to demonstrate a complete Machine Learning workflow, including data preprocessing, exploratory data analysis, feature engineering, model training, and evaluation.

📊 Project Workflow

This project follows a standard Machine Learning pipeline:

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

The dataset used in this project is the California Housing Dataset, which contains housing-related information for districts in California.

Features in the dataset

longitude – Geographic longitude of the house location

latitude – Geographic latitude of the house location

housing_median_age – Median age of houses in the area

total_rooms – Total number of rooms in the district

total_bedrooms – Total number of bedrooms

population – Total population in the district

households – Number of households

median_income – Median income of households

ocean_proximity – Distance from the ocean (categorical feature)

Target Variable

median_house_value – Median house price (value to predict)

🧹 Data Preprocessing

Several preprocessing steps were applied to prepare the dataset for training:

Handling missing values

Filling numerical missing values using mean imputation

Dropping remaining rows containing null values

One-Hot Encoding for categorical feature (ocean_proximity)

Feature Scaling using StandardScaler

These steps help improve model performance and ensure consistent data input.

📊 Exploratory Data Analysis (EDA)

EDA was performed to understand patterns and relationships within the dataset.

Key analyses include:

Correlation Heatmap
Shows relationships between numerical variables and helps identify important features.

Target Variable Distribution
Analyzes the distribution of median_house_value.

Statistical Summary
Provides descriptive statistics such as mean, standard deviation, minimum, and maximum values.

🤖 Machine Learning Model

The model used in this project:

Linear Regression

Linear Regression predicts house prices based on multiple independent variables.
It attempts to model the relationship between input features and the target variable by fitting a linear equation.

📈 Model Evaluation

The model performance is evaluated using several regression metrics:

Mean Absolute Error (MAE)
Measures the average absolute difference between predicted and actual values.

Mean Squared Error (MSE)
Penalizes larger errors more strongly.

Root Mean Squared Error (RMSE)
Square root of MSE, providing error in the same unit as the target variable.

R² Score (Coefficient of Determination)
Indicates how well the model explains variance in the data.

📉 Visualization
Actual vs Predicted Prices

A scatter plot is used to compare:

Actual house prices

Predicted house prices

This visualization helps evaluate how closely the model predictions match the real values.

🛠️ Technologies Used

Python

Pandas

NumPy

Matplotlib

Seaborn

Scikit-learn

Jupyter Notebook
