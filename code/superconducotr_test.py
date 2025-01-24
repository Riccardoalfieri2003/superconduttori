import tensorflow as tf
from keras.models import load_model
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# Load the trained neural network model
model = load_model('superconductor10_model.h5')

# Load the dataset
import sys
import os
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_path)
data = pd.read_csv(project_path+"\\data\\refinedData\\alloy.csv")

# Define the same features used during training
features = [
    'mean_atomic_mass', 
    #'mean_atomic_radius', 
    'mean_Density', 
    'mean_ElectronAffinity', 
    'mean_ThermalConductivity', 
    'mean_Valence', 
    #'entropy_atomic_mass', 
    #'entropy_fie',
    #'range_Density', 'range_Valence'
]

# Extract features (X) and target variable (y)
X_test = data[features]  # Use only the features used during training
y_test = data['critical_temp']  # Target variable

# Scale the features (apply the same scaling as during training)
#scaler = StandardScaler()
#X_test_scaled = scaler.fit_transform(X_test)  # Scaling the test data

# Make predictions using the trained model
#predictions = model.predict(X_test_scaled)


predictions = model.predict(X_test)

# Calculate Mean Absolute Error (MAE) and Mean Squared Error (MSE)
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)

print(f'Mean Absolute Error (MAE): {mae}')
print(f'Mean Squared Error (MSE): {mse}')
"""
# Scatter plot: Actual vs Predicted
plt.scatter(y_test, predictions, color='blue', alpha=0.5)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted')
plt.show()

"""