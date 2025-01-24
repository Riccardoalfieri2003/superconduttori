import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import load_model

# Load your trained model
model = load_model('superconductor10_model.h5')


string_elem = "Au2.041Ag2.404Fe3.354"  # Your test string

string_elem = "Au2.041Ag2.404Fe3.354"
string_elem = "K3.505Br16.334"
string_elem = "Au1.171"
string_elem = "Au2.733Ag1.773Fe3.381K0.095P19O2"

import numpy as np
from scipy.optimize import minimize
import pandas as pd
import os
import re

# Load data
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
elements_df = pd.read_csv(project_path + "\\data\\refinedData\\elements.csv")

# Parse material composition
def parse_material(material_str):
    pattern = r"([A-Z][a-z]*)(\d*\.?\d+)"
    elements = re.findall(pattern, material_str)
    return {symbol: float(quantity) for symbol, quantity in elements}

# Create alloy row
def create_alloy_row(parsed_material):
    atomic_masses = []
    densities = []
    electron_affinities = []
    thermal_conductivities = []
    valences = []
    
    for element, quantity in parsed_material.items():
        if element in elements_df['Symbol'].values:
            element_data = elements_df.loc[elements_df['Symbol'] == element].iloc[0]
            atomic_masses.append(element_data['Atomic Weight'] * quantity)
            densities.append(element_data['Density'] * quantity)
            electron_affinities.append(element_data['ElectronAffinity'] * quantity)
            thermal_conductivities.append(element_data['Thermal Conductivity'] * quantity)
            valences.append(element_data['Valence'] * quantity)
    
    return {
        'mean_atomic_mass': np.mean(atomic_masses),
        'mean_Density': np.mean(densities),
        'mean_ElectronAffinity': np.mean(electron_affinities),
        'mean_ThermalConductivity': np.mean(thermal_conductivities),
        'mean_Valence': np.mean(valences),
    }

# Define the objective function
def objective_function(quantities, elements, model):
    # Create a parsed material dictionary with updated quantities
    parsed_material = {element: quantity for element, quantity in zip(elements, quantities)}
    # Generate the alloy row
    alloy_row = create_alloy_row(parsed_material)
    #print(alloy_row)
    # Predict Tc using the model (replace this with your model's prediction function)
    Tc = abs(model.predict([list(alloy_row.values())])[0])  # Assuming the model takes a list of feature values
    return -Tc  # Negate to maximize Tc



"""
# Optimize the quantities
def optimize_quantities(material_str, model):
    # Parse the material string
    parsed_material = parse_material(material_str)
    elements = list(parsed_material.keys())
    initial_quantities = list(parsed_material.values())
    
    # Define bounds for each quantity (±10% of the initial value)
    bounds = [(q * 0.1, q * 1.9) for q in initial_quantities]
    
    # Run the optimization
    result = minimize(
        objective_function,
        x0=initial_quantities,
        args=(elements, model),
        bounds=bounds,
        method='L-BFGS-B'  # A method suitable for bounded optimization
    )
    
    # Get the optimized quantities
    optimized_quantities = result.x
    optimized_material = {element: quantity for element, quantity in zip(elements, optimized_quantities)}
    return optimized_material
"""


from scipy.optimize import differential_evolution
"""
def optimize_quantities(material_str, model):
    # Parse the material string
    parsed_material = parse_material(material_str)
    elements = list(parsed_material.keys())
    initial_quantities = list(parsed_material.values())

    # Define bounds for optimization
    bounds = [(q * 0.9, q * 1.1) for q in initial_quantities]
    print("Bounds:", bounds)

    # Run the optimization
    result = differential_evolution(
        lambda quantities: objective_function(quantities, elements, model),
        bounds=bounds
    )

    # Get the optimized quantities
    optimized_quantities = result.x
    optimized_material = {element: quantity for element, quantity in zip(elements, optimized_quantities)}

    # Predict Tc for the optimized material
    alloy_row = create_alloy_row(optimized_material)
    predicted_Tc = model.predict([list(alloy_row.values())])[0]  # Assuming model.predict outputs a single value
    print(f"Predicted Temperature (Tc) for Optimized Alloy: {abs(predicted_Tc)}")

    return optimized_material
"""

"""
def optimize_quantities(material_str, model):
    # Parse the material string
    parsed_material = parse_material(material_str)
    elements = list(parsed_material.keys())
    initial_quantities = list(parsed_material.values())

    # Define bounds
    bounds = [(q * 0.9, q * 1.1) for q in initial_quantities]

    # Global Search with optimized parameters
    global_result = differential_evolution(
        lambda quantities: objective_function(quantities, elements, model),
        bounds=bounds,
        maxiter=50,  # Reduce iterations
        popsize=5,   # Reduce population size
        mutation=(0.5, 1),  # Adjust mutation range
        recombination=0.7,  # Adjust recombination rate
        tol=1e-5
    )

    # Local Refinement
    local_result = minimize(
        objective_function,
        x0=global_result.x,
        args=(elements, model),
        bounds=bounds,
        method='L-BFGS-B'
    )

    # Get optimized material
    optimized_quantities = local_result.x
    optimized_material = {element: quantity for element, quantity in zip(elements, optimized_quantities)}

    # Predict Tc for the optimized material
    alloy_row = create_alloy_row(optimized_material)
    predicted_Tc = model.predict([list(alloy_row.values())])[0]
    print(f"Predicted Temperature (Tc) for Optimized Alloy: {abs(predicted_Tc)}")

    return optimized_material
"""

def optimize_quantities(material_str, model):
    # Parse the material string
    parsed_material = parse_material(material_str)
    elements = list(parsed_material.keys())
    initial_quantities = list(parsed_material.values())

    # Define bounds
    bounds = [(q * 0.9, q * 1.1) for q in initial_quantities]

    # Adjust parameters based on the number of elements
    num_elements = len(elements)
    maxiter = max(50 - num_elements * 5, 10)  # Reduce iterations as elements increase
    popsize = max(5 - num_elements, 2)  # Reduce population size as elements increase
    tol = 1e-5 * num_elements  # Increase tolerance for larger element sets

    # Global Search with dynamic parameters
    global_result = differential_evolution(
        lambda quantities: objective_function(quantities, elements, model),
        bounds=bounds,
        maxiter=maxiter,
        popsize=popsize,
        mutation=(0.5, 1),
        recombination=0.7,
        tol=tol
    )

    # Local Refinement
    local_result = minimize(
        objective_function,
        x0=global_result.x,
        args=(elements, model),
        bounds=bounds,
        method='L-BFGS-B'
    )

    # Get optimized material
    optimized_quantities = local_result.x
    optimized_material = {element: quantity for element, quantity in zip(elements, optimized_quantities)}

    # Predict Tc for the optimized material
    alloy_row = create_alloy_row(optimized_material)
    predicted_Tc = model.predict([list(alloy_row.values())])[0]
    print(f"Predicted Temperature (Tc) for Optimized Alloy: {abs(predicted_Tc)}")

    return optimized_material


# Example usage
# Replace `your_model` with your trained model instance
#string_elem = "Fe2.5Ni1.0Co0.5"
#string_elem = "Au2.733Ag1.773Fe3.381K0.095P19O2"
#string_elem = "Au2.733Ag1.773"
#string_elem = "Au2.041Ag2.404Fe3.354Br1.27"
string_elem = "Au2.733Ag1.773Fe3.381K0.095P19O2"


# Parse the material string
parsed_material = parse_material(string_elem)
initial_alloy_row = create_alloy_row(parsed_material)
initial_Tc = model.predict([list(initial_alloy_row.values())])[0]


import time
start_time = time.time()
optimized_material = optimize_quantities(string_elem, model)
end_time = time.time()

print(f"Temperature (Tc) for Unmodified Alloy: {abs(initial_Tc)}")
print("Optimized Material:", optimized_material)

print(f"Time Taken: {end_time - start_time:.2f} seconds")
















"""Predicted Temperature (Tc) for Optimized Alloy: [-64.07359]
Optimized Material: {'Au': 3.0062217976628074, 'Ag': 1.9502988545044442}"""


"""Predicted Temperature (Tc) for Optimized Alloy: [35.63563]
Temperature (Tc) for Unmodified Alloy: [31.480202]
Optimized Material: {'Au': 2.2449293848969307, 'Ag': 2.6439638217382115, 'Fe': 3.689175455563654, 'Br': 1.395183094584475}
Time Taken: 107.68 seconds"""