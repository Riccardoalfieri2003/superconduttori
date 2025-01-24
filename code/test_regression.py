

import joblib
import pandas as pd
import numpy as np
import sys
import os

# Load the element properties from the CSV
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_path)
gan_alloy_dataset = pd.read_csv(project_path+"\\data\\refinedData\\alloy_formation_dataset.csv")
elements = pd.read_csv(project_path+"\\data\\refinedData\\elements.csv")

# Handle missing values in gan_alloy_dataset
gan_alloy_dataset.fillna(value=np.nan, inplace=True)

# Create a dictionary for quick element property lookups (by symbol)
element_properties = elements.set_index('Symbol').to_dict(orient='index')



# Define the dictionary for crystal structure to numeric mapping
crystal_structure_dict = {
    'Base Orthorhombic': 1,
    'Base-centered Monoclinic': 2,
    'Body-centered Cubic': 3,
    'Centered Tetragonal': 4,
    'Face-centered Cubic': 5,
    'Face-centered Orthorhombic': 6,
    'Simple Hexagonal': 7,
    'Simple Monoclinic': 8,
    'Simple Triclinic': 9,
    'Simple Trigonal': 10,
    'Tetrahedral Packing': 11,
    'Unknown': 12
}





# Function to construct input row for the model
def construct_input_row(input_elements, element_properties, max_elements=9):
    row = []
    for element in input_elements:
        if element in element_properties:
            properties = element_properties[element]

            properties['Crystal Structure']=crystal_structure_dict.get(properties['Crystal Structure'])

            # Add numerical properties (AtomicMass, Electronegativity, etc.)
            row.extend([properties['Atomic Weight'], properties['Electronegativity'], properties['Valence'],
                        properties['ElectronAffinity'], properties['Crystal Structure'], properties['Absolute Melting Point']])
        else:
            raise ValueError(f"Element {element} not found in the elements dataset.")
    
    # Fill placeholders for unused elements (if fewer than max_elements)
    for _ in range(len(input_elements), max_elements):
        row.extend([0] * 5)  # Add 5 zeros for the numerical features (instead of NaN)
        row.extend([0])  # Add zero for the crystal structure
    
    return np.array(row).reshape(1, -1)

# Load the trained model
model = joblib.load('regression_model.pkl')

# Create an IterativeImputer (MICE) model
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
mice_imputer = IterativeImputer(max_iter=10, random_state=42)

# Apply the imputer to the input row before prediction
def predict_quantities(input_elements):
    input_row = construct_input_row(input_elements, element_properties)
    #print(f"input_elements shape: {len(input_elements)}")
    #print(f"element_properties shape: {len(element_properties)}")
    #print(f"input_row shape: {input_row.shape}")
    
    # Ensure the input data has the same number of features as the model
    input_row_imputed = mice_imputer.fit_transform(input_row)  # Impute missing values using MICE
    
    # Ensure the input row has the same number of features as the trained model
    if input_row_imputed.shape[1] < model.n_features_in_:
        missing_columns = model.n_features_in_ - input_row_imputed.shape[1]
        input_row_imputed = np.hstack([input_row_imputed, np.zeros((1, missing_columns))])  # Add zeros for missing columns

    predicted_quantities = model.predict(input_row_imputed)
    # Get the first 'n' quantities where 'n' is the length of the input_elements
    n = len(input_elements)
    return predicted_quantities[:n]
"""
# Example usage
input_elements_2 = ['Fe', 'Br']  # Input symbols of elements
predicted_quantities_2 = predict_quantities(input_elements_2)
print(f"Predicted quantities for {input_elements_2}: {predicted_quantities_2}")

input_elements_3 = ['Fe', 'Br', 'O']  # Input symbols of elements
predicted_quantities_3 = predict_quantities(input_elements_3)
print(f"Predicted quantities for {input_elements_3}: {predicted_quantities_3}")

input_elements_4 = ['Fe', 'Br', 'O', 'K']  # Input symbols of elements
predicted_quantities_4 = predict_quantities(input_elements_4)
print(f"Predicted quantities for {input_elements_4}: {predicted_quantities_4}")

input_elements_5 = ['Fe', 'Br', 'O', 'K', 'P']  # Input symbols of elements
predicted_quantities_5 = predict_quantities(input_elements_5)
print(f"Predicted quantities for {input_elements_5}: {predicted_quantities_5}")
"""

def build_material_string(elements, quantities):
    """
    Constructs a string representation of the material in the form 'Fe1.50Br2.01O1.11...'.

    Parameters:
    - elements (list): List of element symbols (e.g., ['Fe', 'Br', 'O']).
    - quantities (list or numpy array): List of corresponding quantities (e.g., [1.5, 2.01, 1.11]).

    Returns:
    - str: A string representing the material composition.
    """
    material_string = ""
    for element, quantity in zip(elements, quantities):
        # Format the quantity to 2 decimal places and append to the element
        material_string += f"{element}{quantity:.3f}"
    return material_string





input_elements_9 = ['Au','Ag','Fe']  # Input symbols of elements
predicted_quantities_9 = predict_quantities(input_elements_9)
#print(f"Predicted quantities for {input_elements_9}: {predicted_quantities_9}")
material_string = build_material_string(input_elements_9, predicted_quantities_9[0])
print(material_string)

input_elements_9 = ['K','Br']  # Input symbols of elements
predicted_quantities_9 = predict_quantities(input_elements_9)
#print(f"Predicted quantities for {input_elements_9}: {predicted_quantities_9}")
material_string = build_material_string(input_elements_9, predicted_quantities_9[0])
print(material_string)


input_elements_9 = ['Au']  # Input symbols of elements
predicted_quantities_9 = predict_quantities(input_elements_9)
#print(f"Predicted quantities for {input_elements_9}: {predicted_quantities_9}")
material_string = build_material_string(input_elements_9, predicted_quantities_9[0])
print(material_string)


input_elements_9 = ['Au','Ag','Fe', 'O', 'K']  # Input symbols of elements
predicted_quantities_9 = predict_quantities(input_elements_9)
#print(f"Predicted quantities for {input_elements_9}: {predicted_quantities_9}")
material_string = build_material_string(input_elements_9, predicted_quantities_9[0])
print(material_string)

