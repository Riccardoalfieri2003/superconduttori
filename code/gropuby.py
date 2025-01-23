import pandas as pd
import os

"""
# Load the element properties from the CSV
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
periodic_table_path = os.path.join(project_path, "data", "refinedData", "elements.csv")
periodic_table = pd.read_csv(periodic_table_path)

# Group elements by Crystal Structure
grouped = periodic_table.groupby('Crystal Structure')

# Save each group to a separate CSV file (optional)
#output_dir = os.path.join(project_path, "data", "grouped_by_crystal_structure")
#os.makedirs(output_dir, exist_ok=True)

#for structure, group in grouped:
    # Save each group to a separate CSV file
#    group.to_csv(os.path.join(output_dir, f"{structure}_elements.csv"), index=False)

# Display groups (optional)
for structure, group in grouped:
    print(f"Crystal Structure: {structure}")
    print(group)
    print("-" * 40)"""


# Load the element properties from the CSV
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
periodic_table_path = os.path.join(project_path, "data", "refinedData", "elements.csv")
periodic_table = pd.read_csv(periodic_table_path)

# Get unique crystal structures
unique_structures = periodic_table['Crystal Structure'].unique()

# Display the unique structures
print("Unique Crystal Structures:")
for structure in unique_structures:
    print(structure)