import numpy as np
import torch

from keras.models import load_model

# Load the pre-trained model
model = load_model('superconductor10_model.h5')

# Assuming `model` is your trained neural network
def critical_temperature(quantities):
    # Convert input to the required shape for the model
    input_array = np.array(quantities).reshape(1, -1)  # Reshape for batch size
    prediction = model.predict(input_array, verbose=0)  # Suppress verbose output
    return prediction[0, 0]  # Assuming the model outputs a single value


from scipy.optimize import minimize

# Example alloy quantities
alloy = {"K": 3.505, "Br": 16.334}
initial = list(alloy.values())

# Bounds for each quantity (±5%)
bounds = [(0.95 * q, 1.05 * q) for q in initial]

# Define the optimization objective (maximize critical temperature)
result = minimize(lambda x: -critical_temperature(x), initial, bounds=bounds)

# Optimal quantities
optimal_quantities = result.x
print("Optimal Quantities:", optimal_quantities)
print("Maximum Critical Temperature:", -result.fun)