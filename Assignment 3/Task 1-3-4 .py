import numpy as np
import pandas as pd
import math
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import boxcox
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score

# Load your dataset into a DataFrame 
df = pd.read_csv("/content/po2_data.csv")
# Define the feature columns (X) and target columns (y)
X = df.drop(['motor_updrs', 'total_updrs'], axis=1)
y_motor_updrs = df['motor_updrs']
y_total_updrs = df['total_updrs']
missing_columns = X.columns[X.isna().any()]
print("Columns with missing values:", missing_columns)

# Spliting the data into training (50%) and test (50%) sets
X_train, X_test, y_motor_train, y_motor_test, y_total_train, y_total_test = train_test_split(X, y_motor_updrs, y_total_updrs, test_size=0.5, random_state=42)
#Applying the log transformation
X_train_log = np.log1p(X_train)
X_test_log = np.log1p(X_test)
# Calculate correlation coefficients
correlation_matrix = X_train_log.corr()

# Create a heatmap of correlation coefficients
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()
# Create a SimpleImputer to replace NaN values with the mean
imputer = SimpleImputer(strategy='mean')
# Fit the scaler to the training data and transform the training and testing data
X_train_std = imputer.fit_transform(X_train_log)
X_test_std = imputer.transform(X_test_log)

# Apply the Box-Cox transformation to the target variables
y_motor_train_bc, lam_motor_updrs = boxcox(y_motor_train)
y_total_train_bc, lam_total_updrs = boxcox(y_total_train)
# Initializing Linear Regression models for Motor UPDRS and TOTAL UPDRS
motor_updrs_model = LinearRegression()
total_updrs_model = LinearRegression()
# Training the models on training data
motor_updrs_model.fit(X_train_std, y_motor_train_bc)
total_updrs_model.fit(X_train_std, y_total_train_bc)

# Making predictions on the test set
motor_updrs_predictions_bc = motor_updrs_model.predict(X_test_std)
total_updrs_predictions_bc = total_updrs_model.predict(X_test_std)
# Reverse the Box-Cox transformation to obtain predictions in the original scale
motor_updrs_predictions = np.power(motor_updrs_predictions_bc, 1 / lam_motor_updrs)
total_updrs_predictions = np.power(total_updrs_predictions_bc, 1 / lam_total_updrs)
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