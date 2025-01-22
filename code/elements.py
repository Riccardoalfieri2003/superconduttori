import pandas as pd
import numpy as np
import re

df=pd.read_csv("data/elements_data.csv")


df['Density'] = df['Density'].str.split(' ').str[0].astype(float)

df['Absolute Melting Point'] = df['Absolute Melting Point'].str.replace(' K', '', regex=False).astype(float)
df['Absolute Boiling Point'] = df['Absolute Boiling Point'].str.replace(' K', '', regex=False).astype(float)

df['Critical Pressure'] = (
    df['Critical Pressure']
    .str.extract(r'\(([\d.]+) Atm\)', expand=False)  # Extract the Atm value
    .astype(float)  # Convert to float
)
df['Critical Temperature'] = df['Critical Temperature'].str.replace(' K', '', regex=False).astype(float)

df['Specific Heat'] = df['Specific Heat'].str.split(' ').str[0].astype(float)


df['Thermal Conductivity'] = df['Thermal Conductivity'].str.split(' ').str[0].astype(float)
# Apply the transformation
df['Thermal Expansion'] = df['Thermal Expansion'] \
    .str.split(' ').str[0] \
    .str.replace('×', 'e') \
    .str.replace('e10','e')\
    .astype(float)



df['Molar Volume'] = df['Molar Volume']\
    .str.split(' ').str[0] \
    .str.replace('×', 'e') \
    .str.replace('e10','e')\
    .astype(float)


df['Electronegativity'] = df['Electronegativity'].astype(float)
df['ElectronAffinity'] = df['ElectronAffinity'].str.split(' ').str[0].astype(float)

df['Valence'] = df['Valence'].astype(float)
df['IonizationEnergy_Mean'] = df['Ionization Energies'].apply(
    lambda x: np.mean([float(i.replace(' kJ/mol', '').strip()) for i in str(x).split(',') if isinstance(i, str)]) 
    if isinstance(x, str) else np.nan
)



def parse_electron_config(config):
    # Regex to extract orbitals (handle cases like 7s2, 6d1, etc.)
    orbitals = re.findall(r'(\d+[spdf]\d+)', config)
    
    # Initialize counts dictionary for s, p, d, f orbitals
    counts = {'sOrb': 0, 'pOrb': 0, 'dOrb': 0, 'fOrb': 0}
    
    for orbital in orbitals:
        # Extract the principal quantum number (number) and the orbital type (s, p, d, f)
        match = re.match(r'(\d+)([spdf])(\d+)', orbital)
        
        if match:
            number = int(match.group(1))  # Principal quantum number (e.g., 7 from 7s2)
            type_ = match.group(2)  # Orbital type (s, p, d, f)
            count = int(match.group(3))  # Number of electrons in the orbital (e.g., 2 from 7s2)
            
            # Increment the appropriate orbital count
            if type_ == 's':
                counts['sOrb'] += count
            elif type_ == 'p':
                counts['pOrb'] += count
            elif type_ == 'd':
                counts['dOrb'] += count
            elif type_ == 'f':
                counts['fOrb'] += count
    
    return counts


parsed_features = df['Electron Configuration'].apply(parse_electron_config).apply(pd.Series)
df = pd.concat([df, parsed_features], axis=1)



def parse_quantum_numbers(term):
    # Extract principal quantum number (n)
    n = int(term[0])
    
    # Map orbital angular momentum letters to numbers
    l_map = {'S': 0, 'P': 1, 'D': 2, 'F': 3, 'G': 4, 'H': 5, 'I': 6, 'K': 7, 'L': 8, 'M': 9}
    orbital_letter = term[1]
    l = l_map.get(orbital_letter, None)
    
    if l is None:
        raise ValueError(f"Unknown orbital angular momentum letter: {orbital_letter}")
    
    # Extract magnetic quantum number (m)
    # If there is a '/' in the term, it's a fraction (e.g., '7/2'), take the numerator as m
    if '/' in term[2:]:
        m = int(term[2:].split('/')[0])
    else:
        m = int(term[2:])
    
    # Extract spin quantum number and multiplicity
    if '/' in term[2:]:
        spin = term[2:].split('/')[1]
        spin_multiplicity = int(spin)
    else:
        spin_multiplicity = 2  # Default spin multiplicity for terms like '1S0'

    return {'n Qn': n, 'l Qn': l, 'm Qn': m, 'spin Qn': spin_multiplicity}

parsed_features = df['Quantum Numbers'].apply(parse_quantum_numbers).apply(pd.Series)
df = pd.concat([df, parsed_features], axis=1)



df['Electrical Conductivity'] = df['Electrical Conductivity']\
    .str.split(' ').str[0] \
    .str.replace('×', 'e') \
    .str.replace('e10','e')\
    .astype(float)

df['Resistivity'] = df['Resistivity']\
    .str.split(' ').str[0] \
    .str.replace('×', 'e') \
    .str.replace('e10','e')\
    .astype(float)

df['Superconducting Point']=df['Superconducting Point'].astype(float)

df['Magnetic Type']=df['Magnetic Type']
df['Mass Magnetic Susceptibility'] = df['Mass Magnetic Susceptibility']\
    .str.split(' ').str[0] \
    .str.replace('×', 'e') \
    .str.replace('e10','e')\
    .astype(float)

df['Molar Magnetic Susceptibility'] = df['Molar Magnetic Susceptibility']\
    .str.split(' ').str[0] \
    .str.replace('×', 'e') \
    .str.replace('e10','e')\
    .astype(float)

df['Crystal Structure']=df['Crystal Structure']
df['Atomic Number']=df['Atomic Number']



    #"sOrb", "pOrb", "dOrb", "fOrb",
    #"n Qn", "l Qn", "m Qn", "spin Qn",

selected_columns=[
    "Atomic Number",
    "Name",
    "Symbol",
    "Density",
    "Absolute Melting Point",
    "Absolute Boiling Point",
    "Critical Pressure",
    "Specific Heat",
    "Thermal Conductivity",
    "Thermal Expansion",
    "Molar Volume",
    "Electronegativity",
    "ElectronAffinity",
    "Valence",
    "IonizationEnergy_Mean",
    "Electrical Conductivity",
    "Resistivity",
    "Superconducting Point",
    "Magnetic Type",
    "Mass Magnetic Susceptibility",
    "Molar Magnetic Susceptibility",
    "Crystal Structure"
]
cleaned_df = df[selected_columns].replace("", np.nan)

print(cleaned_df)

cleaned_df.to_csv('new_dataset.csv', index=False)