"""import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib

import sys
import os

# Load the element properties from the CSV
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_path)
gan_alloy_dataset = pd.read_csv(project_path+"\\data\\refinedData\\gan_alloy_dataset.csv")

# Handle missing values
gan_alloy_dataset.fillna(value=np.nan, inplace=True)

# Extract features and target
features = []
targets = []

max_elements = 9  # Maximum number of elements in an alloy
for i in range(1, max_elements + 1):
    features.extend([
        f'AtomicMass{i}', f'Electronegativity{i}', f'Valence{i}', 
        f'ElectronAffinity{i}', f'Absolute Melting Point{i}'
    ])
    targets.append(f'Quantity{i}')

# Handle categorical feature (Crystal Structure)
crystal_features = [f'Crystal Structure{i}' for i in range(1, max_elements + 1)]
ohe = OneHotEncoder(sparse=False)
crystal_encoded = ohe.fit_transform(gan_alloy_dataset[crystal_features].fillna('Unknown'))

# Combine numerical and encoded features
numerical_data = gan_alloy_dataset[features].fillna(0)
X = np.hstack([numerical_data, crystal_encoded])

# Combine target quantities
y = gan_alloy_dataset[targets].fillna(0).values


# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R² Score: {r2}")

# Save the trained model
joblib.dump(model, 'regression_model.pkl')
print("Model saved successfully!")
"""


"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
import joblib
import sys
import os

# Load the element properties from the CSV
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_path)
gan_alloy_dataset = pd.read_csv(project_path+"\\data\\refinedData\\gan_alloy_dataset.csv")

# Handle missing values
gan_alloy_dataset.fillna(value=np.nan, inplace=True)

# Extract features and target
features = []
targets = []

max_elements = 9  # Maximum number of elements in an alloy
for i in range(1, max_elements + 1):
    features.extend([
        f'AtomicMass{i}', f'Electronegativity{i}', f'Valence{i}', 
        f'ElectronAffinity{i}', f'Absolute Melting Point{i}'
    ])
    targets.append(f'Quantity{i}')

# Handle categorical feature (Crystal Structure)
crystal_features = [f'Crystal Structure{i}' for i in range(1, max_elements + 1)]
ohe = OneHotEncoder(sparse=False)
crystal_encoded = ohe.fit_transform(gan_alloy_dataset[crystal_features].fillna('Unknown'))

# Combine numerical and encoded features
numerical_data = gan_alloy_dataset[features].fillna(0)
X = np.hstack([numerical_data, crystal_encoded])

# Combine target quantities
y = gan_alloy_dataset[targets].fillna(0).values

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print the shapes of the datasets
print(f"numerical data shape: {numerical_data.shape}")
print(f"crystal_encoded shape: {crystal_encoded.shape}")
print(f"y shape: {y.shape}")
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")



# Initialize and train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R² Score: {r2}")

# Save the trained model
joblib.dump(model, 'regression_model.pkl')
print("Model saved successfully!")
"""


# Import necessary libraries
import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load the element properties from the CSV
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_path)
gan_alloy_dataset = pd.read_csv(project_path+"\\data\\refinedData\\alloy_formation_dataset.csv")

# Handle missing values
gan_alloy_dataset.fillna(value=np.nan, inplace=True)

# Extract features and target
features = []
targets = []




# max_elements is set to 9 as per your original setup
max_elements = 9

# Initialize the list for features and targets
features = []
targets = []


# Add atomic properties and quantities for each element
for i in range(1, max_elements + 1):
    features.extend([f'AtomicMass{i}', f'Electronegativity{i}', f'Valence{i}', 
                     f'ElectronAffinity{i}',f'Crystal Structure{i}', f'Absolute Melting Point{i}'])
    targets.append(f'Quantity{i}')





# Combine numerical and encoded features
numerical_data = gan_alloy_dataset[features].fillna(0)
#X = np.hstack([numerical_data, crystal_encoded])
X = np.hstack([numerical_data])

# Combine target quantities
y = gan_alloy_dataset[targets].fillna(0).values

# Check the shapes of the datasets
#print(f"numerical data shape: {numerical_data.shape}")
#print(f"crystal_encoded shape: {crystal_encoded.shape}")
#print(f"y shape: {y.shape}")
#print(f"X shape: {X.shape}")
#print(f"y shape: {y.shape}")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check the shapes after splitting
#print(f"X_train shape: {X_train.shape}")
#print(f"X_test shape: {X_test.shape}")
#print(f"y_train shape: {y_train.shape}")
#print(f"y_test shape: {y_test.shape}")


# Initialize and train the model
model = RandomForestRegressor(n_estimators=500, random_state=42)
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate the model
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R² Score: {r2}")

# Save the trained model
joblib.dump(model, 'regression_model5.pkl')
print("Model saved successfully!")
