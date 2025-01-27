import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from joblib import load

# Set up project path
import os
import sys
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_path)

# Load the data
data = pd.read_csv(project_path + "\\data\\refinedData\\alloyDatasetPlus2.csv")

# Definire le feature e il target
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
target = 'critical_temp'

X = data[features]
y = data[target]

# Suddividere i dati in training e testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Gestire i valori mancanti con imputazione
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Standardizzare le feature
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

# Caricare il modello salvato
model = load('random_forest_model.joblib')

# Effettuare le previsioni
y_pred = model.predict(X_test_scaled)

# Calcolare le metriche di validazione
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Stampare le metriche
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R2): {r2:.2f}")

# Generare il grafico della retta di regressione
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', label='Previsioni', alpha=0.6)
#plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='y = x (Ideale)')

# Calcolare la retta di regressione
coefficients = np.polyfit(y_test, y_pred, 1)  # Regressione lineare sui dati previsti
polynomial = np.poly1d(coefficients)
plt.plot(y_test, polynomial(y_test), color='red', label=f"y = {coefficients[0]:.2f}x + {coefficients[1]:.2f}")

# Personalizzare il grafico
plt.xlabel("Valori Reali (y_test)")
plt.ylabel("Valori Predetti (y_pred)")
plt.title("Grafico della Regressione")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
