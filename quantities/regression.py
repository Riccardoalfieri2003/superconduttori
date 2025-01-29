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
alloy_dataset = pd.read_csv(project_path+"\\data\\refinedData\\alloy_formation_dataset.csv")

# Handle missing values
#alloy_dataset.fillna(value=np.nan, inplace=True)

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
    features.extend([f'AtomicMass{i}', f'Electronegativity{i}', 
                     f'Valence{i}', f'ElectronAffinity{i}',
                     f'Crystal Structure{i}', f'Absolute Melting Point{i}'])
    targets.append(f'Quantity{i}')


#Imputazione dei valori mancanti con la media per le caratteristiche
numerical_data = alloy_dataset[features].fillna(alloy_dataset[features].mean())

# Imputazione dei valori mancanti con la media per i target (quantità)
y = alloy_dataset[targets].fillna(alloy_dataset[targets].mean()).values

X = np.hstack([numerical_data])


# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
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
