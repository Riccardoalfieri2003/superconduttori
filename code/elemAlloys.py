import pandas as pd
import re
import sys
import os

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

alloy_df['parsed_material'] = alloy_df['material'].apply(parse_material)

# Combine element properties with alloy composition
def create_alloy_row(parsed_material):
    row = {}
    for i, (element, quantity) in enumerate(parsed_material.items()):
        if element in elements_df['Symbol'].values:
            element_data = elements_df.loc[elements_df['Symbol'] == element].iloc[0]
            row[f'Element{i+1}'] = element
            row[f'Quantity{i+1}'] = quantity
            row[f'AtomicMass{i+1}'] = element_data['Atomic Weight']
            row[f'Electronegativity{i+1}'] = element_data['Electronegativity']
            row[f'Valence{i+1}'] = element_data['Valence']
            row[f'ElectronAffinity{i+1}'] = element_data['ElectronAffinity']
            row[f'Crystal Structure{i+1}'] = element_data['Crystal Structure']
            row[f'Absolute Melting Point{i+1}'] = element_data['Absolute Melting Point']
            # Add other properties as needed
        else:
            print(f"Warning: Element {element} not found in elements.csv")
    return row

# Generate rows for each alloy
alloy_rows = alloy_df['parsed_material'].apply(create_alloy_row)
alloy_expanded_df = pd.DataFrame(alloy_rows.tolist())

# Save the dataset
alloy_expanded_df.to_csv("gan_alloy_dataset.csv", index=False)