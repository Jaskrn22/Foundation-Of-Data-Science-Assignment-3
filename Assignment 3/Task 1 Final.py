import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import r2_score
# Read dataset into a DataFrame
data = pd.read_csv("/content/po2_data.csv")
# Defining the feature columns (x) and target columns (y)
x = data.drop(['motor_updrs', 'total_updrs'], axis=1)
y_motor_updrs = data['motor_updrs']
y_total_updrs = data['total_updrs']
missing_columns = x.columns[x.isna().any()]
print("Columns with missing values:", missing_columns)
# Spliting the data into training (60%) and test (40%) sets
x_train, x_test, y_motor_train, y_motor_test, y_total_train, y_total_test = train_test_split(x, y_motor_updrs, y_total_updrs, test_size=0.4, random_state=42)

# Initializing Linear Regression models for Motor UPDRS and TOTAL UPDRS
motor_updrs_model = LinearRegression()
total_updrs_model = LinearRegression()

# Training the models on training data
motor_updrs_model.fit(x_train, y_motor_train)
total_updrs_model.fit(x_train, y_total_train)

# Making predictions on the test set
motor_updrs_predictions = motor_updrs_model.predict(x_test)
total_updrs_predictions = total_updrs_model.predict(x_test)
# Print the intercept and coefficient learned by the linear regression model
print("Intercept: ", motor_updrs_model.intercept_)
print("Coefficient: ", motor_updrs_model.coef_)
print("Intercept: ", total_updrs_model.intercept_)
print("Coefficient: ", total_updrs_model.coef_)

# Showing the predicted values of (y) next to the actual values of (y)
data_motor_pred = pd.DataFrame({"Actual": y_motor_test, "Predicted": motor_updrs_predictions})
data_total_pred = pd.DataFrame({"Actual": y_total_test, "Predicted": total_updrs_predictions})
print(data_motor_pred)
print(data_total_pred)
# Compute standard performance metrics of the linear regression:

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
# Visualizing the results of Predicted values using scatter plots of actual vs. predicted scores

# Plotting actual vs. predicted Motor UPDRS scores
plt.scatter(y_motor_test, motor_updrs_predictions)
plt.xlabel("Actual Motor UPDRS Score")
plt.ylabel("Predicted Motor UPDRS Score")
plt.title("Actual vs. Predicted Motor UPDRS Scores")
plt.show()

# Plotting actual vs. predicted Total UPDRS scores
plt.scatter(y_total_test, total_updrs_predictions)
plt.xlabel("Actual Total UPDRS Score")
plt.ylabel("Predicted Total UPDRS Score")
plt.title("Actual vs. Predicted Total UPDRS Scores")
plt.show()
# Printing the performance of the model
print("Motor UPDRS Model:")
print("MAEMOTOR: ", mae_motor)
print("MSEMOTOR: ", mse_motor) 
print("RMSEMOTOR: ", rmse_motor)
print("RMSEMOTOR (Normalised): ", rmse_motor_norm)
print("R2_Score(Motor_Updrs):",motor_updrs_r2)
print("Adjusted Motor R-squared:", adjusted_motor_r_squared)
print("-" * 50)
print("\nTotal UPDRS Model:")
print("MAETOTAL: ", mae_total)
print("MSETOTAL: ", mse_total)
print("RMSETOTAL: ", rmse_total)
print("RMSETOTAL (Normalised): ", rmse_total_norm)
print("R2_Score(Total_Updrs):",total_updrs_r2)
print("Adjusted Total R-squared:", adjusted_total_r_squared)
print("\n")