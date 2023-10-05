import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import r2_score

# Load the dataset
data = pd.read_csv('po2_data.csv')

# Define the feature columns (X) and target columns (y)
X = data.drop(['motor_updrs', 'total_updrs'], axis=1)
y_motor_updrs = data['motor_updrs']
y_total_updrs = data['total_updrs']

# Create a list of different train-test split ratios
split_ratios = [0.5, 0.6, 0.7, 0.8]

# Initialize Linear Regression models
motor_updrs_model = LinearRegression()
total_updrs_model = LinearRegression()

for split_ratio in split_ratios:
    # Split the data into training and test sets
    X_train, X_test, y_motor_train, y_motor_test, y_total_train, y_total_test = train_test_split(X, y_motor_updrs, y_total_updrs, test_size=(1 - split_ratio), random_state=42)

    # Train the models
    motor_updrs_model.fit(X_train, y_motor_train)
    total_updrs_model.fit(X_train, y_total_train)

    # Make predictions on the test set
    motor_updrs_predictions = motor_updrs_model.predict(X_test)
    total_updrs_predictions = total_updrs_model.predict(X_test)

    # Evaluate the models
    # Mean Absolute Error
    mae_motor = metrics.mean_absolute_error(y_motor_test, motor_updrs_predictions)
    mae_total = metrics.mean_absolute_error(y_total_test, total_updrs_predictions)
    # Mean Squared Error
    mse_motor = metrics.mean_squared_error(y_motor_test, motor_updrs_predictions)
    mse_total = metrics.mean_squared_error(y_total_test, total_updrs_predictions)
    # Root Mean Square Error
    rmse_motor =  math.sqrt(metrics.mean_squared_error(y_motor_test, motor_updrs_predictions))
    rmse_total =  math.sqrt(metrics.mean_squared_error(y_total_test, total_updrs_predictions))
    # Normalised Root Mean Square Error
    y_motor_max = y_motor_test.max()
    y_motor_min = y_motor_test.min()
    y_total_max = y_total_test.max()
    y_total_min = y_total_test.min()
    rmse_motor_norm = rmse_motor / (y_motor_max - y_motor_min)
    rmse_total_norm = rmse_total / (y_total_max - y_total_min)
    # R2 score
    motor_updrs_r2 = r2_score(y_motor_test, motor_updrs_predictions)
    total_updrs_r2 = r2_score(y_total_test, total_updrs_predictions)
    # adjusted R2 score
    # Number of observations (data points)
    n = 5876
    # Number of predictors (independent variables)
    p = 17
    # Calculate adjusted R-squared
    adjusted_motor_r_squared = 1 - (1 - motor_updrs_r2) * (n - 1) / (n - p - 1)
    adjusted_total_r_squared = 1 - (1 - total_updrs_r2) * (n - 1) / (n - p - 1)

    print(f"Train-Test Split Ratio: {int(split_ratio*100)}% - {int((1-split_ratio)*100)}%")
    print("Motor UPDRS Model:")
    print("MAEMOTOR: ", mae_motor)
    print("MSEMOTOR: ", mse_motor) 
    print("RMSEMOTOR: ", rmse_motor)
    print("RMSEMOTOR (Normalised): ", rmse_motor_norm)
    print("R2_Score(Motor_Updrs):",motor_updrs_r2)
    print("Adjusted Motor R-squared:", adjusted_motor_r_squared)
    print("\nTotal UPDRS Model:")
    print("MAETOTAL: ", mae_total)
    print("MSETOTAL: ", mse_total)
    print("RMSETOTAL: ", rmse_total)
    print("RMSETOTAL (Normalised): ", rmse_total_norm)
    print("R2_Score(Total_Updrs):",total_updrs_r2)
    print("Adjusted Total R-squared:", adjusted_total_r_squared)
    print("-" * 50)
    print("\n")