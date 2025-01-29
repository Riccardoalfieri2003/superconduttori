import pandas as pd

# Load the datasets
elements_df = pd.read_csv("elements.csv")
alloy_df = pd.read_csv("alloy.csv")

import re

def parse_material(material_str):
    # Use regular expression to extract element symbols and their quantities
    pattern = r"([A-Z][a-z]*)(\d*\.?\d+)"
    elements = re.findall(pattern, material_str)
    
    # Convert the list of tuples into a dictionary
    element_dict = {symbol: float(quantity) for symbol, quantity in elements}
    return element_dict

# Apply the function to parse the material column
alloy_df['parsed_material'] = alloy_df['material'].apply(parse_material)


# Create a dictionary of element properties for quick lookup
element_properties = elements_df.set_index('Symbol').to_dict(orient='index')

def aggregate_alloy_properties(parsed_material):
    alloy_properties = {}
    
    # List of valid numeric properties to aggregate
    valid_properties = [
        "Density", "Melting Point", "Boiling Point", "Critical Pressure", "Specific Heat",
        "Thermal Conductivity", "Thermal Expansion", "Molar Volume", "Electronegativity", 
        "ElectronAffinity", "Valence", "IonizationEnergy_Mean", "Electrical Conductivity", 
        "Resistivity", "Superconducting Point", "Mass Magnetic Susceptibility", 
        "Molar Magnetic Susceptibility"
    ]
    
    # Loop through each element in the alloy and retrieve its properties
    for element, quantity in parsed_material.items():
        if element in element_properties:
            element_data = element_properties[element]
            
            # Aggregate the valid properties by multiplying with the quantity of the element in the alloy
            for prop, value in element_data.items():
                if prop in valid_properties and isinstance(value, (int, float)):  # Ensure value is numeric and valid
                    if prop not in alloy_properties:
                        alloy_properties[prop] = 0
                    alloy_properties[prop] += value * quantity
                #elif prop not in valid_properties:
                    # Optionally, print a message for excluded properties (e.g., Magnetic Type, Crystal Structure)
                    #print(f"Excluding non-numeric property: {prop} for element {element}")
    
    return alloy_properties
# Apply the function to aggregate properties for each alloy
alloy_df['aggregated_properties'] = alloy_df['parsed_material'].apply(aggregate_alloy_properties)

# Define the properties we want to include in the feature vector
selected_properties = [
    "Density", "Melting Point", "Boiling Point", "Critical Pressure", "Specific Heat",
    "Thermal Conductivity", "Thermal Expansion", "Molar Volume", "Electronegativity", 
    "ElectronAffinity", "Valence", "IonizationEnergy_Mean", "Electrical Conductivity", 
    "Resistivity", "Superconducting Point", "Magnetic Type", "Mass Magnetic Susceptibility", 
    "Molar Magnetic Susceptibility", "Crystal Structure"
]

# Function to convert aggregated properties to a feature vector
def create_feature_vector(aggregated_properties):
    feature_vector = []
    
    # Extract the selected properties from the aggregated properties
    for prop in selected_properties:
        feature_vector.append(aggregated_properties.get(prop, 0))  # Default to 0 if the property is missing
    
    return feature_vector

# Apply the function to create the feature vectors
alloy_df['feature_vector'] = alloy_df['aggregated_properties'].apply(create_feature_vector)


# Create a DataFrame with the feature vectors and the target variable
features_df = pd.DataFrame(alloy_df['feature_vector'].tolist())
features_df['critical_temp'] = alloy_df['critical_temp']

features_df.to_csv("features_with_critical_temp.csv", index=False)