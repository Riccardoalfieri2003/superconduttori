import pandas as pd
import re
import sys
import os

string_elem="Au2.041Ag2.404Fe3.354"


project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_path)
#print(sys.path)

# Load datasets
alloy_df = pd.read_csv(project_path+"\\data\\refinedData\\alloy.csv")
elements_df = pd.read_csv(project_path+"\\data\\refinedData\\elements.csv")


# Parse material composition
def parse_material(material_str):
    pattern = r"([A-Z][a-z]*)(\d*\.?\d+)"
    elements = re.findall(pattern, material_str)
    return {symbol: float(quantity) for symbol, quantity in elements}



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


# Modify the create_alloy_row function to assign numeric values to crystal structures
def create_alloy_row(parsed_material):
    row = {}
    for i, (element, quantity) in enumerate(parsed_material.items()):
        if element in elements_df['Symbol'].values:
            element_data = elements_df.loc[elements_df['Symbol'] == element].iloc[0]
            row[f'Element{i+1}'] = element
            row[f'Quantity{i+1}'] = quantity

            row[f'AtomicMass{i+1}'] = element_data['Atomic Weight']
            row[f'AtomicRadius{i+1}'] = element_data['Valence']
            #row[f'Electronegativity{i+1}'] = element_data['Electronegativity']
            row[f'Valence{i+1}'] = element_data['Valence']
            row[f'ElectronAffinity{i+1}'] = element_data['ElectronAffinity']
            
            # Assign the numeric value for the crystal structure using the dictionary
            #crystal_structure = element_data['Crystal Structure']
            #row[f'Crystal Structure{i+1}'] = crystal_structure_dict.get(crystal_structure, 12)  # Default to 12 if not found
            
            row[f'Absolute Melting Point{i+1}'] = element_data['Absolute Melting Point']
            # Add other properties as needed
        else:
            print(f"Warning: Element {element} not found in elements.csv")
    return row




# Parse the string material
parsed_material = parse_material(string_elem)

# Generate the row for the parsed material using create_alloy_row
alloy_row = create_alloy_row(parsed_material)


print(alloy_row)


#alloy_expanded_df = pd.DataFrame(alloy_row.tolist())



# Save the dataset
#alloy_expanded_df.to_csv("alloy_formation_dataset.csv", index=False)



