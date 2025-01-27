import numpy as np
import pandas as pd
import re
import sys
import os

string_elem="Au2.041Ag2.404Fe3.354"


project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_path)
#print(sys.path)

#elements_df = pd.read_csv(project_path+"\\data\\refinedData\\elements.csv")
elements_df = pd.read_csv(project_path+"\\data\\elements_data.csv")
unique_m_df  = pd.read_csv(project_path+"\\data\\unique_m.csv")


import pandas as pd
import numpy as np
import re

# Load the original dataset
#unique_m_df = pd.read_csv('unique_m.csv')

# Parse material composition
def parse_material(material_str):
    pattern = r"([A-Z][a-z]*)(\d*\.?\d+)"
    elements = re.findall(pattern, material_str)
    return {symbol: float(quantity) for symbol, quantity in elements}

elements_df['IonizationEnergy_Mean'] = elements_df['Ionization Energies'].apply(
    lambda x: np.mean([float(i.replace(' kJ/mol', '').strip()) for i in str(x).split(',') if isinstance(i, str)]) 
    if isinstance(x, str) else np.nan
)

"""
# Modify the create_alloy_row function to assign numeric values to crystal structures
def create_alloy_row(parsed_material, elements_df):

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

            element_data = elements_df.loc[elements_df['Symbol'] == element].iloc[0]
            row[f'Element{i+1}'] = element
            row[f'Quantity{i+1}'] = quantity

            row[f'AtomicMass{i+1}'] = element_data['Atomic Weight'] * quantity
            atomic_masses.append(row[f'AtomicMass{i+1}'])

            try: row[f'AtomicRadius{i+1}'] = float(element_data['Atomic Radius'].split(' ')[0]) * quantity
            except: row[f'AtomicRadius{i+1}']=np.nan
            atomic_radius.append(row[f'AtomicRadius{i+1}'])

            row[f'Density{i+1}'] = float(element_data['Density'].split(' ')[0]) * quantity
            densities.append(row[f'Density{i+1}'])

            row[f'Valence{i+1}'] = element_data['Valence'] * quantity
            valences.append(row[f'Valence{i+1}'])

            row[f'Electronegativity{i+1}'] = element_data['Electronegativity'] * quantity
            electronegativity.append(row[f'Electronegativity{i+1}'])

            row[f'ElectronAffinity{i+1}'] = float(element_data['ElectronAffinity'].split(' ')[0]) * quantity
            electron_affinities.append(row[f'ElectronAffinity{i+1}'])

            row[f'IonizingEnergyMean{i+1}'] = float(element_data['IonizationEnergy_Mean'] *quantity)
            ion_energy_mean.append(row[f'IonizingEnergyMean{i+1}'])

            try:
                row[f'ElectricalConductivity{i+1}'] = float(element_data['Electrical Conductivity'].split(' ')[0].replace('×', 'e').replace('e10', 'e')) * quantity * 0.00001
            except:
                row[f'ElectricalConductivity{i+1}'] = 0
            elec_cond.append(row[f'ElectricalConductivity{i+1}'])

            try:
                row[f'Resistivity{i+1}'] = float(element_data['Resistivity'].split(' ')[0].replace('×', 'e').replace('e10', 'e')) * 100000000 * quantity
            except:
                row[f'Resistivity{i+1}'] = 0
            resistivity.append(row[f'Resistivity{i+1}'])

            try:
                row[f'MassMagneticSusceptibility{i+1}'] = float(element_data['Mass Magnetic Susceptibility'].split(' ')[0].replace('×', 'e').replace('e10', 'e')) * 100000000 * quantity
            except:
                row[f'MassMagneticSusceptibility{i+1}'] = 0
            MassMagneticSusceptibility.append(row[f'MassMagneticSusceptibility{i+1}'])

            row[f'Absolute Melting Point{i+1}'] = float(element_data['Absolute Melting Point'].split(' ')[0])
            amp.append(row[f'Absolute Melting Point{i+1}'])

            try:
                row[f'Thermal Conductivity{i+1}'] = float(element_data['Thermal Conductivity'].split(' ')[0]) * quantity
            except:
                row[f'Thermal Conductivity{i+1}'] = 0
            thermal_conductivities.append(row[f'Thermal Conductivity{i+1}'])

        else:
            print(f"Warning: Element {element} not found in elements.csv")

    return {
        'mean_atomic_mass': np.mean(atomic_masses),
        'mean_atomic_radius':np.mean(atomic_radius),
        'mean_Density': np.mean(densities),
        'mean_ElectronAffinity': np.mean(electron_affinities),
        'mean_ThermalConductivity': np.mean(thermal_conductivities),
        'mean_Valence': np.mean(valences),
        'mean_Electronegativity': np.mean(electronegativity),
        'mean_IonizingEnergies': np.mean(ion_energy_mean),
        'mean_AbsoluteMeltingPoint': np.mean(amp),
        'mean_ElectricalConductivity': np.mean(elec_cond),
        'mean_Resistivity': np.mean(resistivity),
        'mean_MassMagneticSusceptibility': np.mean(MassMagneticSusceptibility)
    }"""


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

            row[f'IonizingEnergyMean{i+1}'] = float(element_data['IonizationEnergy_Mean'] * percentage)
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

    """return {
        'mean_atomic_mass': np.mean(atomic_masses),
        'mean_atomic_radius': np.mean(atomic_radius),
        'mean_Density': np.mean(densities),
        'mean_ElectronAffinity': np.mean(electron_affinities),
        'mean_ThermalConductivity': np.mean(thermal_conductivities),
        'mean_Valence': np.mean(valences),
        'mean_Electronegativity': np.mean(electronegativity),
        'mean_IonizingEnergies': np.mean(ion_energy_mean),
        'mean_AbsoluteMeltingPoint': np.mean(amp),
        'mean_ElectricalConductivity': np.mean(elec_cond),
        'mean_Resistivity': np.mean(resistivity),
        'mean_MassMagneticSusceptibility': np.mean(MassMagneticSusceptibility)
    }"""

    return {
        'mean_atomic_mass': (atomic_masses),
        'mean_atomic_radius':(atomic_radius),
        'mean_Density': (densities),
        'mean_ElectronAffinity':(electron_affinities),
        'mean_ThermalConductivity': (thermal_conductivities),
        'mean_Valence': (valences),
        'mean_Electronegativity': (electronegativity),
        'mean_IonizingEnergies': (ion_energy_mean),
        'mean_AbsoluteMeltingPoint': (amp),
        'mean_ElectricalConductivity': (elec_cond),
        'mean_Resistivity': (resistivity),
        'mean_MassMagneticSusceptibility': (MassMagneticSusceptibility)
    }



# Create a new dataset based on unique_m.csv
new_data = []

for index, row in unique_m_df.iterrows():
    material_str = row['material']  # Assuming the material column is named 'material'
    parsed_material = parse_material(material_str)
    alloy_row = create_alloy_row(parsed_material, elements_df)
    alloy_row['critical_temp'] = row['critical_temp']  # Add the critical temperature from the original dataset
    alloy_row['material'] = material_str  # Add the material string
    new_data.append(alloy_row)

# Create a new DataFrame with the calculated values
new_df = pd.DataFrame(new_data)

# Save the new dataset
new_df.to_csv('alloyDatasetPlus2.csv', index=False)
