import torch





import os
import sys
import pandas as pd
# Load the element properties from the CSV
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_path)



periodic_table = pd.read_csv(project_path+"\\data\\refinedData\\elements.csv")

# List of the features you want to keep for each element
features = ['Atomic Weight', 'Electronegativity', 'Valence', 'ElectronAffinity', 'Crystal Structure', 'Absolute Melting Point']


# Get unique crystal structures
unique_structures = periodic_table['Crystal Structure'].unique()

# Create a mapping of crystal structures to indices
structure_to_index = {structure: idx for idx, structure in enumerate(unique_structures)}

# One-hot encode the crystal structures
def encode_crystal_structure(crystal_structure):
    encoded = [0] * len(unique_structures)
    if crystal_structure in structure_to_index:
        encoded[structure_to_index[crystal_structure]] = 1
    return encoded

# Add the one-hot encoded crystal structure to the DataFrame
periodic_table['Crystal Structure'] = periodic_table['Crystal Structure'].apply(encode_crystal_structure)



# Create the dictionary with element number as the key and the selected features as the value
element_properties = {}

# Iterate over each row in the DataFrame
for index, row in periodic_table.iterrows():
    element_number = row['Atomic Number']  # Assuming 'Element' column has the element number (1 to 118)
    
    # Create a dictionary for the element with the specified features
    element_info = {feature: row[feature] for feature in features}
    
    # Add the element and its info to the dictionary
    element_properties[element_number] = element_info

# Check the structure
#for element, properties in element_properties.items():
#    print(f"{element}: {properties}")




# Create a mapping of element symbols to atomic numbers
symbol_to_atomic_number = {row['Symbol']: row['Atomic Number'] for index, row in periodic_table.iterrows()}

def get_element_properties(elements):
    properties = []
    for element in elements:
        # Get the atomic number from the element symbol
        element_number = symbol_to_atomic_number.get(element)
        
        if element_number is not None:
            # Retrieve the element's properties using the atomic number
            if element_number in element_properties:
                element_data = element_properties[element_number]
                
                # Flatten the feature vector: Add numerical features + one-hot encoded structure
                flat_features = [
                    element_data['Atomic Weight'],
                    element_data['Electronegativity'],
                    element_data['Valence'],
                    element_data['ElectronAffinity'],
                    *element_data['Crystal Structure'],  # Flatten the one-hot encoded structure
                    element_data['Absolute Melting Point']
                ]
                properties.append(flat_features)
            else:
                print(f"Warning: Element {element} (atomic number {element_number}) not found in element_properties.")
        else:
            print(f"Warning: Element symbol {element} not found in symbol_to_atomic_number.")
    
    return properties  # Returns a list of flat numerical feature vectors











# Assuming the models are saved as 'generator_model.pth' and 'discriminator_model.pth'
#discriminator_path = os.path.join('alloy_discriminator.pth')
#generator_path = os.path.join('alloy_generator.pth')

from classes import ElementGenerator
from classes import ElementDiscriminator


latent_dim = 128  # Dimension of the latent space
feature_dim = 54  # Number of features (adjust based on dataset)
output_dim = 9    # Maximum number of quantities (corresponding to 9 elements)
num_elements = 118  # Change this to the actual number of elements you have

# Now initialize the generator with the required num_elements argument
generator = ElementGenerator(latent_dim=latent_dim, feature_dim=feature_dim, output_dim=output_dim, num_elements=num_elements)
discriminator = ElementDiscriminator()



# Load the state dictionaries
generator.load_state_dict(torch.load('alloy_generator.pth'))
discriminator.load_state_dict(torch.load('alloy_discriminator.pth'))

# Set the models to evaluation mode
generator.eval()
discriminator.eval()




def process_input(elements):
    # Get a list of feature vectors for the elements
    feature_vectors = get_element_properties(elements)

    
    
    # Flatten the list of feature vectors into a single list
    flattened_features = [value for vector in feature_vectors for value in vector]

    print(flattened_features)
    
    # Convert to a PyTorch tensor
    return torch.tensor(flattened_features).float()




# Example list of elements
user_input = ['Al', 'Cu', 'Fe']  # Ensure this is a list of full element symbols
input_tensor = process_input(user_input)




# Pass the input through the generator to get the predicted quantities
with torch.no_grad():  # Disable gradient computation for inference
    #predicted_quantities = generator(input_tensor)
    batch_size = input_tensor.size(0)  # Get the batch size from the input tensor
    element_indices, predicted_quantities = generator(batch_size)

# Assuming the model output is a tensor of predicted quantities
print("Predicted Quantities for the Alloy:")
print(predicted_quantities)

# Assuming predicted_quantities is a tensor with the quantities of each element
def display_output(predicted_quantities, elements):
    for element, quantities in zip(elements, predicted_quantities):
        quantities_str = ", ".join(f"{q:.4f}" for q in quantities)  # Format each quantity to 4 decimal places
        print(f"{element}: [{quantities_str}]")  # Display all quantities for the element


# Display the predicted quantities for the user
display_output(predicted_quantities, user_input)
