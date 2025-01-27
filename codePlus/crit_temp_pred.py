import pandas as pd
import numpy as np

from joblib import load
# Load the saved model
model = load('random_forest_model.joblib')



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
elements_df = pd.read_csv(project_path + "\\data\\elements_data.csv")

# Parse material composition
def parse_material(material_str):
    pattern = r"([A-Z][a-z]*)(\d*\.?\d+)"
    elements = re.findall(pattern, material_str)
    return {symbol: float(quantity) for symbol, quantity in elements}


elements_df['IonizationEnergy_Mean'] = elements_df['Ionization Energies'].apply(
    lambda x: np.mean([float(i.replace(' kJ/mol', '').strip()) for i in str(x).split(',') if isinstance(i, str)]) 
    if isinstance(x, str) else np.nan
)


# Create alloy row
def create_alloy_row(parsed_material, elements_df):
    # Calculate the total quantity of the alloy
    total_quantity = sum(parsed_material.values())

    atomic_masses = []
    atomic_radius = []
    densities = []
    electron_affinities = []
    electronegativity = []
    ion_energy_mean = []
    thermal_conductivities = []
    valences = []
    elec_cond = []
    resistivity = []
    MassMagneticSusceptibility = []
    amp = []
    supCondPoint = []

    row = {}
    for i, (element, quantity) in enumerate(parsed_material.items()):
        if element in elements_df['Symbol'].values:
            # Calculate the percentage of the element in the alloy
            percentage = quantity / total_quantity

            element_data = elements_df.loc[elements_df['Symbol'] == element].iloc[0]
            row[f'Element{i+1}'] = element
            row[f'Quantity{i+1}'] = quantity

            row[f'AtomicMass{i+1}'] = element_data['Atomic Weight'] * percentage
            atomic_masses.append(row[f'AtomicMass{i+1}'])

            try: row[f'AtomicRadius{i+1}'] = float(element_data['Atomic Radius'].split(' ')[0]) * percentage
            except: row[f'AtomicRadius{i+1}'] = np.nan
            atomic_radius.append(row[f'AtomicRadius{i+1}'])

            row[f'Density{i+1}'] = float(element_data['Density'].split(' ')[0]) * percentage
            densities.append(row[f'Density{i+1}'])

            row[f'Valence{i+1}'] = element_data['Valence'] * percentage
            valences.append(row[f'Valence{i+1}'])

            row[f'Electronegativity{i+1}'] = element_data['Electronegativity'] * percentage
            electronegativity.append(row[f'Electronegativity{i+1}'])

            row[f'ElectronAffinity{i+1}'] = float(element_data['ElectronAffinity'].split(' ')[0]) * percentage
            electron_affinities.append(row[f'ElectronAffinity{i+1}'])

            row[f'IonizingEnergyMean{i+1}'] = float(element_data['IonizationEnergy_Mean']) * percentage
            ion_energy_mean.append(row[f'IonizingEnergyMean{i+1}'])

            try:
                row[f'ElectricalConductivity{i+1}'] = float(element_data['Electrical Conductivity'].split(' ')[0].replace('×', 'e').replace('e10', 'e')) * percentage * 0.00001
            except:
                row[f'ElectricalConductivity{i+1}'] = 0
            elec_cond.append(row[f'ElectricalConductivity{i+1}'])

            try:
                row[f'Resistivity{i+1}'] = float(element_data['Resistivity'].split(' ')[0].replace('×', 'e').replace('e10', 'e')) * 100000000 * percentage
            except:
                row[f'Resistivity{i+1}'] = 0
            resistivity.append(row[f'Resistivity{i+1}'])

            try:
                row[f'MassMagneticSusceptibility{i+1}'] = float(element_data['Mass Magnetic Susceptibility'].split(' ')[0].replace('×', 'e').replace('e10', 'e')) * 100000000 * percentage
            except:
                row[f'MassMagneticSusceptibility{i+1}'] = 0
            MassMagneticSusceptibility.append(row[f'MassMagneticSusceptibility{i+1}'])

            row[f'Absolute Melting Point{i+1}'] = float(element_data['Absolute Melting Point'].split(' ')[0])
            amp.append(row[f'Absolute Melting Point{i+1}'])

            try:
                row[f'Thermal Conductivity{i+1}'] = float(element_data['Thermal Conductivity'].split(' ')[0]) * percentage
            except:
                row[f'Thermal Conductivity{i+1}'] = 0
            thermal_conductivities.append(row[f'Thermal Conductivity{i+1}'])

        else:
            print(f"Warning: Element {element} not found in elements.csv")


    return {
        'mean_atomic_mass': atomic_masses[0],
        'mean_atomic_radius':(atomic_radius[0]),
        'mean_Density': (densities[0]),
        'mean_ElectronAffinity':(electron_affinities[0]),
        'mean_ThermalConductivity': (thermal_conductivities[0]),
        'mean_Valence': (valences[0]),
        'mean_Electronegativity': (electronegativity[0]),
        'mean_IonizingEnergies': (ion_energy_mean[0]),
        'mean_AbsoluteMeltingPoint': (amp[0]),
        'mean_ElectricalConductivity': (elec_cond[0]),
        'mean_Resistivity': (resistivity[0]),
        'mean_MassMagneticSusceptibility': (MassMagneticSusceptibility[0])
    }





# Define the objective function
def objective_function(quantities, elements, model):
    # Create a parsed material dictionary with updated quantities
    parsed_material = {element: quantity for element, quantity in zip(elements, quantities)}
    # Generate the alloy row
    alloy_row = create_alloy_row(parsed_material,elements_df)
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
    bounds = [(q * 0.8, q * 1.2) for q in initial_quantities]

    # Adjust parameters based on the number of elements
    #num_elements = len(elements)
    #maxiter = max(50 - num_elements * 5, 10)  # Reduce iterations as elements increase
    #popsize = max(5 - num_elements, 2)  # Reduce population size as elements increase
    #tol = 1e-5 * num_elements  # Increase tolerance for larger element sets

    #maxiter=50,  # Reduce iterations
    #popsize=5,
    
    # Global Search with dynamic parameters
    """global_result = differential_evolution(
        lambda quantities: objective_function(quantities, elements, model),
        bounds=bounds,
        maxiter=100,
        popsize=10,
        #mutation=(0.5, 1),
        mutation=(0.8, 1.3),
        recombination=0.9,
        tol=1e-5
    )"""
    
    global_result = differential_evolution(
        lambda quantities: objective_function(quantities, elements, model),
        bounds=bounds,
        maxiter=50,
        popsize=5,
        mutation=(0.5, 1),
        recombination=0.7,
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
    alloy_row = create_alloy_row(optimized_material,elements_df)
    predicted_Tc = model.predict([list(alloy_row.values())])[0]
    print(f"Predicted Temperature (Tc) for Optimized Alloy: {abs(predicted_Tc)}")

    return optimized_material


# Example usage
# Replace `your_model` with your trained model instance
#string_elem = "Fe2.5Ni1.0Co0.5"
#string_elem = "Au2.733Ag1.773Fe3.381K0.095P19O2"
#string_elem = "Au2.733Ag1.773"
#string_elem = "Au2.041Ag2.404Fe3.354Br1.27"
#string_elem = "Au2.733Ag1.773Fe3.381K0.095P19O2"

string_elem = "Au2.041Ag2.404Fe3.354Br1.27In0.3K1.79"
#string_elem = "Au1"

# Parse the material string
parsed_material = parse_material(string_elem)
#print(parsed_material)
initial_alloy_row = create_alloy_row(parsed_material,elements_df)
#print(initial_alloy_row)

import time


start_time = time.time()
initial_Tc = model.predict([list(initial_alloy_row.values())]).flatten()[0]
end_time = time.time()

print(f"Temperature (Tc) for Unmodified Alloy: {abs(initial_Tc)}")
print("Material: ", parsed_material)

print(f"Time Taken: {end_time - start_time:.5f} seconds")

print()


start_time = time.time()
optimized_material = optimize_quantities(string_elem, model)
end_time = time.time()

print(f"Temperature (Tc) for Unmodified Alloy: {abs(initial_Tc)}")
print("Optimized Material:", optimized_material)

print(f"Time Taken: {end_time - start_time:.5f} seconds")
















"""Predicted Temperature (Tc) for Optimized Alloy: [-64.07359]
Optimized Material: {'Au': 3.0062217976628074, 'Ag': 1.9502988545044442}"""


"""Predicted Temperature (Tc) for Optimized Alloy: [35.63563]
Temperature (Tc) for Unmodified Alloy: [31.480202]
Optimized Material: {'Au': 2.2449293848969307, 'Ag': 2.6439638217382115, 'Fe': 3.689175455563654, 'Br': 1.395183094584475}
Time Taken: 107.68 seconds"""