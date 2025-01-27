import pandas as pd
import numpy as np


import sys
import os

# Load the element properties from the CSV
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_path)
train_df = pd.read_csv(project_path+"\\data\\train.csv")
elements = pd.read_csv(project_path+"\\data\\elements_data.csv")
unique_df= pd.read_csv(project_path+"\\data\\unique_m.csv")


selected_columns_train=[

    #éropietà atomiche
    "mean_atomic_mass",
    #"mean_atomic_radius",
    "mean_Density",
    "mean_Electronegativity",
    "mean_ElectronAffinity",
    "mean_Valence",
    "mean_IonizingEnergies",

    #Propietà fisiche
    "mean_AbsoluteMeltingPoint" #"mean_FusionHeat",
    "mean_ThermalConductivity",
    #"mean_ThermalExpansion",
    "mean_SuperconductivePoint",

    "mean_ElectricalConductivity",
    #"mean_ElectricalResistivity",
    "mean_MassMagneticSusceptibility",
    
    #variabile importante
    "critical_temp"
]






selected_columns_unique=["material"]
print(selected_columns_unique)


cleaned_df=train_df[selected_columns_train].replace("", np.nan)
merged_df = cleaned_df.reset_index(drop=True).merge(unique_df[selected_columns_unique].reset_index(drop=True), left_index=True, right_index=True, how="left")

print(merged_df)

merged_df.to_csv('alloyPlus.csv', index=False)