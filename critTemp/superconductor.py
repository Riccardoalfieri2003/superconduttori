"""import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout


import sys
import os
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_path)
data = pd.read_csv(project_path+"\\data\\refinedData\\alloyDatasetPlus2.csv")

features = [
    'mean_atomic_mass',
    'mean_atomic_radius',
    'mean_Density',
    'mean_ElectronAffinity',
    'mean_ThermalConductivity',
    'mean_Valence',
    'mean_Electronegativity',
    'mean_IonizingEnergies',
    'mean_AbsoluteMeltingPoint',
    'mean_ElectricalConductivity',
    'mean_Resistivity',
    'mean_MassMagneticSusceptibility'
]

X = data[features]
y = data['critical_temp']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.layers import BatchNormalization

# Preprocessing
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model architecture
model = Sequential([
    Dense(256, activation='relu', input_dim=X_train_scaled.shape[1]),
    Dropout(0.2),
    BatchNormalization(),
    Dense(128, activation='relu'),
    Dropout(0.2),
    BatchNormalization(),
    Dense(64, activation='relu'),
    Dense(1)  # Single output for regression
])

# Compile model
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

# Early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(
    X_train_scaled, y_train,
    validation_data=(X_test_scaled, y_test),
    epochs=50, batch_size=8, verbose=1,
    callbacks=[early_stopping]
)

# Evaluate the model
test_loss, test_mae = model.evaluate(X_test_scaled, y_test)
print(f"Test MAE: {test_mae}")

# Predict on new data
y_pred = model.predict(X_test_scaled)"""




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
data = pd.read_csv(project_path+"\\data\\refinedData\\alloyDatasetPlus2.csv")

features = [
    'mean_atomic_mass',
    'mean_atomic_radius',
    'mean_Density',
    'mean_ElectronAffinity',
    'mean_ThermalConductivity',
    'mean_Valence',
    'mean_Electronegativity',
    'mean_IonizingEnergies',
    'mean_AbsoluteMeltingPoint',
    'mean_ElectricalConductivity',
    'mean_Resistivity',
    'mean_MassMagneticSusceptibility'
]

X = data[features]
y = data['critical_temp']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

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

model.save('superconductorPlus2_model.h5')

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
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.impute import SimpleImputer

import sys
import os

# Set up project path
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_path)

# Load the data
data = pd.read_csv(project_path + "\\data\\refinedData\\alloyDatasetPlus2.csv")

# Define features and target
features = [
    'mean_atomic_mass',
    'mean_atomic_radius',
    'mean_Density',
    'mean_ElectronAffinity',
    'mean_ThermalConductivity',
    'mean_Valence',
    'mean_Electronegativity',
    'mean_IonizingEnergies',
    'mean_AbsoluteMeltingPoint',
    'mean_ElectricalConductivity',
    'mean_Resistivity',
    'mean_MassMagneticSusceptibility'
]

X = data[features]
y = data['critical_temp']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Scale the features (optional for Random Forest, but good practice)
scaler = StandardScaler()

# Create an imputer to fill missing values with the mean
imputer = SimpleImputer(strategy='mean')

# Fit the imputer on training data and transform both training and test data
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Scale the data after imputation
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

# Initialize the Random Forest Regressor
rf_model = RandomForestRegressor( n_estimators=100, random_state=42 )

# Train the model
rf_model.fit(X_train_scaled, y_train)

# Evaluate the model
y_pred = rf_model.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"Test MAE: {mae}")
print(f"Test MSE: {mse}")

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance)


from joblib import dump

# Save the trained model to a file
dump(rf_model, 'random_forest_model.joblib')

print("Model saved as 'random_forest_model.joblib'")


"""
# Initialize the Random Forest Regressor
rf_model = RandomForestRegressor(
    n_estimators=100,  # Number of trees
    max_depth=None,    # Allow trees to grow fully
    random_state=42,   # For reproducibility
    n_jobs=-1          # Use all available cores
)
"""