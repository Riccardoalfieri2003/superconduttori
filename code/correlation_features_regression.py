import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

# Assuming you have the input data and predicted quantities
# X: The feature data, y_pred: The predicted quantities

import sys
import os

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_path)
data = pd.read_csv(project_path+"\\data\\refinedData\\alloy_formation_dataset.csv")


#print(data.columns)


# Extract the features (X) - All columns except the quantities (target variables)
feature_columns = [
    'AtomicMass1', 'Electronegativity1', 'Valence1', 'ElectronAffinity1', 'Crystal Structure1', 'Absolute Melting Point1',
    'AtomicMass2', 'Electronegativity2', 'Valence2', 'ElectronAffinity2', 'Crystal Structure2', 'Absolute Melting Point2',
    'AtomicMass3', 'Electronegativity3', 'Valence3', 'ElectronAffinity3', 'Crystal Structure3', 'Absolute Melting Point3',
    'AtomicMass4', 'Electronegativity4', 'Valence4', 'ElectronAffinity4', 'Crystal Structure4', 'Absolute Melting Point4',
    'AtomicMass5', 'Electronegativity5', 'Valence5', 'ElectronAffinity5', 'Crystal Structure5', 'Absolute Melting Point5',
    'AtomicMass6', 'Electronegativity6', 'Valence6', 'ElectronAffinity6', 'Crystal Structure6', 'Absolute Melting Point6',
    'AtomicMass7', 'Electronegativity7', 'Valence7', 'ElectronAffinity7', 'Crystal Structure7', 'Absolute Melting Point7',
    'AtomicMass8', 'Electronegativity8', 'Valence8', 'ElectronAffinity8', 'Crystal Structure8', 'Absolute Melting Point8',
    'AtomicMass9', 'Electronegativity9', 'Valence9', 'ElectronAffinity9', 'Crystal Structure9', 'Absolute Melting Point9'
]

X = data[feature_columns]  # Features

# Extract the target variables (y) - Quantities for each element
target_columns = ['Quantity1', 'Quantity2', 'Quantity3', 'Quantity4', 'Quantity5', 'Quantity6', 'Quantity7', 'Quantity8', 'Quantity9']
y = data[target_columns]  # Target variables (quantities)

# Now X contains the feature columns, and y contains the target quantities for each element
#print(X.head())  # Print the first few rows of X to verify
#print(y.head())  # Print the first few rows of y to verify


import matplotlib.pyplot as plt
import seaborn as sns


model = joblib.load('regression_model.pkl')


# Use X as the test features (X_test)
X_test = X  # Since you're using the entire dataset for prediction

# Select a specific quantity (e.g., Quantity1) to compare
target_variable = 'Quantity1'  # Change this to the quantity you want to evaluate (Quantity1, Quantity2, etc.)

# Impute missing values in y
y_imputer = SimpleImputer(strategy='mean')
y_imputed = pd.Series(y_imputer.fit_transform(y[target_variable].values.reshape(-1, 1)).flatten(), index=y.index)

# Now split the data again
X_train, X_test, y_train, y_test = train_test_split(X, y_imputed, test_size=0.2, random_state=42)

# Create an imputer and pipeline as before
imputer = SimpleImputer(strategy='mean')
model_pipeline = make_pipeline(imputer, model)

# Fit the pipeline on the training data
model_pipeline.fit(X_train, y_train)

X_test, y_test = X_test.align(y_test, join='inner', axis=0)
print(len(X_test), len(y_test))  # Should be the same length


# Predict using the pipeline
predicted_quantities = model_pipeline.predict(X_test)

# Fit a linear regression line to the data (y_test vs predicted_quantities)
slope, intercept = np.polyfit(y_test, predicted_quantities, 1)

# Create the regression line based on the slope and intercept
regression_line = slope * y_test + intercept

# Now plot
plt.scatter(y_test, predicted_quantities, color='blue', alpha=0.5, label='Data Points')
plt.plot(y_test, regression_line, color='red', label=f'Linear Fit: y = {slope:.2f}x + {intercept:.2f}')
plt.xlabel('Actual Quantities')
plt.ylabel('Predicted Quantities')
plt.title(f'Actual vs Predicted for {target_variable}')
plt.legend()
plt.show()



from sklearn.metrics import mean_absolute_error, mean_squared_error

# Calculate MAE and MSE using y_test (actual values) and predicted_quantities
mae = mean_absolute_error(y_test, predicted_quantities)
mse = mean_squared_error(y_test, predicted_quantities)

print(f'Mean Absolute Error (MAE): {mae}')
print(f'Mean Squared Error (MSE): {mse}')