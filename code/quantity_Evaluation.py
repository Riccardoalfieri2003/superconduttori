import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from joblib import load
import sys
import os

# Caricamento del dataset
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
gan_alloy_dataset = pd.read_csv(os.path.join(project_path, "data", "refinedData", "alloy_formation_dataset.csv"))

# Gestione dei valori mancanti
gan_alloy_dataset.fillna(value=np.nan, inplace=True)

# Definizione delle feature e dei target
max_elements = 9
features = []
targets = []

for i in range(1, max_elements + 1):
    features.extend([f'AtomicMass{i}', f'Electronegativity{i}', f'Valence{i}', f'ElectronAffinity{i}',
                     f'Crystal Structure{i}', f'Absolute Melting Point{i}'])
    targets.append(f'Quantity{i}')

# Preparazione delle feature numeriche
numerical_data = gan_alloy_dataset[features].fillna(0)
X = np.hstack([numerical_data])

# Preparazione dei target
y = gan_alloy_dataset[targets].fillna(0).values

# Caricamento del modello salvato
model = load('regression_model.pkl')

# Effettuare le previsioni
y_pred = model.predict(X)

# Flatten delle matrici per ottenere un singolo vettore per tutte le quantità
y_flat = y.flatten()
y_pred_flat = y_pred.flatten()

# Calcolo delle metriche globali
mae = mean_absolute_error(y_flat, y_pred_flat)
mse = mean_squared_error(y_flat, y_pred_flat)
rmse = np.sqrt(mse)
r2 = r2_score(y_flat, y_pred_flat)

print("\nMetriche globali:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R2): {r2:.2f}")

# Rimuovere l'outlier (l'elemento con il valore più alto)
max_value_index = np.argmax(y_flat)
y_flat_filtered = np.delete(y_flat, max_value_index)
y_pred_flat_filtered = np.delete(y_pred_flat, max_value_index)

# Generare il grafico globale della regressione senza l'outlier
plt.figure(figsize=(8, 6))
plt.scatter(y_flat_filtered, y_pred_flat_filtered, color='blue', label='Previsioni', alpha=0.6)

# Calcolare la retta di regressione globale
coefficients = np.polyfit(y_flat_filtered, y_pred_flat_filtered, 1)
polynomial = np.poly1d(coefficients)
plt.plot(y_flat_filtered, polynomial(y_flat_filtered), color='green', label=f"y = {coefficients[0]:.2f}x + {coefficients[1]:.2f}")


# Personalizzazione del grafico
plt.xlabel("Valori Reali (y)")
plt.ylabel("Valori Predetti (y_pred)")
plt.title("Grafico della Regressione Globale (Senza Outlier)")
plt.legend()
plt.grid(alpha=0.3)
plt.show()