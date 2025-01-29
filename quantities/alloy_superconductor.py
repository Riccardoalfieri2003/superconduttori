import pandas as pd
import numpy as np

train_df=pd.read_csv("data/train.csv")

selected_columns_train=[

    #éropietà atomiche
    "mean_atomic_mass",
    "std_atomic_mass",
    "mean_atomic_radius",
    "std_atomic_radius",
    "mean_Density",
    "std_Density",
    "mean_ElectronAffinity",
    "mean_FusionHeat",
    "mean_ThermalConductivity",
    "mean_Valence",

    #Pesi e medie geometroche
    "wtd_mean_atomic_mass", "wtd_mean_fie",
    "gmean_atomic_radius", "gmean_Density",

    #Diversità e spread
    "entropy_atomic_mass", "entropy_fie",
    "range_Density", "range_Valence",
    "std_ThermalConductivity", "std_Valence",

    #variabile importante
    "critical_temp"
]



unique_df=pd.read_csv("data/unique_m.csv")
selected_columns_unique=["material"]


cleaned_df=train_df[selected_columns_train].replace("", np.nan)
merged_df = cleaned_df.reset_index(drop=True).merge(unique_df[selected_columns_unique].reset_index(drop=True), left_index=True, right_index=True, how="left")

print(merged_df)

merged_df.to_csv('merged_df.csv', index=False)