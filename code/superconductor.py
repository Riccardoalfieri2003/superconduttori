import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout

import sys
import os
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_path)
data = pd.read_csv(project_path+"\\data\\refinedData\\alloy.csv")


# Load your dataset
# Assuming `data` is a pandas DataFrame with all the features and target variable
# Replace `data` with your actual DataFrame
features = [
    'mean_atomic_mass', 
    #'std_atomic_mass', 
    #'mean_atomic_radius', 
    #'std_atomic_radius',
    'mean_Density', 
    #'std_Density', 
    'mean_ElectronAffinity', 
    #'mean_FusionHeat',
    'mean_ThermalConductivity', 
    'mean_Valence', 
    #'wtd_mean_atomic_mass', 'wtd_mean_fie',
    #'gmean_atomic_radius', 'gmean_Density', 
    #'entropy_atomic_mass', 
    #'entropy_fie',
    #'range_Density', 'range_Valence'
    #'std_ThermalConductivity', 'std_Valence'
]

X = data[features]
y = data['critical_temp']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
#X_train_scaled = scaler.fit_transform(X_train)
#X_test_scaled = scaler.transform(X_test)

X_train_scaled = X_train
X_test_scaled = X_test

# Build the neural network
model = Sequential([
    Dense(128, activation='relu', input_dim=X_train_scaled.shape[1]),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1)  # Single output for regression
])



# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the model
history = model.fit(
    X_train_scaled, y_train,
    validation_data=(X_test_scaled, y_test),
    epochs=50, batch_size=8, verbose=1
)

model.save('superconductor10_model.h5')

# Evaluate the model
test_loss, test_mae = model.evaluate(X_test_scaled, y_test)
print(f"Test MAE: {test_mae}")

# Predict on new data
y_pred = model.predict(X_test_scaled)


"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout

import sys
import os
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_path)
data = pd.read_csv(project_path+"\\data\\refinedData\\alloy.csv")


features = [
    'mean_atomic_mass', 
    'mean_Density', 
    'mean_ElectronAffinity', 
    'mean_ThermalConductivity', 
    'mean_Valence'
]

X = data[features]
y = data['critical_temp']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = Sequential([
    Dense(128, activation='relu', input_dim=X_train.shape[1]),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1, activation='softplus')  # Apply Softplus to the output layer
])

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the model
history = model.fit(
    X_train_scaled, y_train,
    validation_data=(X_test_scaled, y_test),
    epochs=50, batch_size=16, verbose=1
)


model.save('superconductor20_model.h5')

# Evaluate the model
test_loss, test_mae = model.evaluate(X_test, y_test)
print(f"Test MAE: {test_mae}")

# Predict on new data
y_pred = model.predict(X_test)"""