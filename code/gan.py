"""import torch
import random


# Generator model: Generates up to 9 elements
class ElementGenerator(torch.nn.Module):
    def __init__(self):
        super(ElementGenerator, self).__init__()

    def forward(self, batch_size):
        # Generate a random number of elements (1 to 9)
        num_elements = random.randint(1, 9)

        # Randomly select 'num_elements' unique elements from the dictionary
        elements = random.sample(range(1, 119), num_elements)
        
        return torch.tensor(elements)  # Return as tensor for further processing

# Instantiate the generator
element_generator = ElementGenerator()

# Generate elements for a batch (let's say batch_size = 1 for now)
elements = element_generator(batch_size=1)
print(f"Generated elements: {elements}")






import pandas as pd
import sys
import os
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_path)

# Load the CSV file into a DataFrame
df = pd.read_csv(project_path+"\\data\\refinedData\\elements.csv")

# List of the features you want to keep for each element
features = ['Name','Atomic Weight', 'Electronegativity', 'Valence', 'ElectronAffinity', 'Crystal Structure', 'Absolute Melting Point']

# Create the dictionary with element number as the key and the selected features as the value
element_properties = {}

# Iterate over each row in the DataFrame
for index, row in df.iterrows():
    element_number = row['Atomic Number']  # Assuming 'Element' column has the element number (1 to 118)
    
    # Create a dictionary for the element with the specified features
    element_info = {feature: row[feature] for feature in features}
    
    # Add the element and its info to the dictionary
    element_properties[element_number] = element_info

#print(element_properties)


# Manually map the generated elements to their properties
def get_element_properties(elements):
    properties = []
    for element in elements:
        element_data = element_properties.get(element.item(), None)
        if element_data:
            properties.append(element_data)
    return properties

# Get the properties of the generated elements
properties = get_element_properties(elements)
for property in properties:
    print(property)

"""
import torch
torch.autograd.set_detect_anomaly(True)


import random
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import sys
import os

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



# Manually map the generated elements to their properties
def get_element_properties(elements):
    properties = []
    for element_tensor in elements:
        # Convert the tensor to a scalar (assuming it's a single value per element)
        element_index = int(element_tensor.argmax().item())  # Use argmax to find the most probable element

        # Retrieve the element symbol based on 'Atomic Number'
        if element_index in element_properties:
            element_data = element_properties[element_index]
            properties.append(element_data)
        else:
            print(f"Warning: Element index {element_index} not found in element_properties.")
    return properties




# Generator model: Generates up to 9 elements and their quantities
"""
class ElementGenerator(nn.Module):
    def __init__(self, latent_dim, feature_dim):
        super(ElementGenerator, self).__init__()
        self.latent_dim = latent_dim
        self.feature_dim = feature_dim
        # Define a linear layer to map latent space to features
        self.latent_to_features_layer = nn.Linear(latent_dim, feature_dim)

    def latent_to_features(self, batch_size):
        # Generate latent vectors
        latent_vectors = torch.randn(batch_size, self.latent_dim)
        # Transform latent vectors into features
        features = self.latent_to_features_layer(latent_vectors)
        return features

    def forward(self, batch_size):
        x = self.latent_to_features(batch_size)
        
        # Randomly choose the number of elements to generate (between 1 and 9)
        num_elements = torch.randint(1, 10, (1,)).item()
        
        # Generate unique random indices for elements (1 to 118)
        element_indices = torch.randint(1, 119, (num_elements,)).tolist()  # Random numbers between 1 and 118

        # Slice the tensor to get element features and quantities
        elements = x[:, :num_elements]  # Features for elements
        quantities = x[:, -9:]  # Quantities
        quantities = quantities[:, :num_elements]  # Match quantities with elements

        # Generate properties for the elements
        element_properties_generated = []
        for idx in element_indices:
            if idx in element_properties:
                element_properties_generated.append(element_properties[idx])
            else:
                print(f"Warning: Element index {idx} not found in element_properties.")
                element_properties_generated.append(None)  # Default or empty properties

        # Normalize quantities (optional)
        quantities = torch.nn.functional.softmax(quantities, dim=1)
        
        return elements, quantities
"""

"""
class ElementGenerator(nn.Module):
    def __init__(self, latent_dim, feature_dim, output_dim):
        super(ElementGenerator, self).__init__()
        self.latent_dim = latent_dim
        self.feature_dim = feature_dim
        self.output_dim = output_dim

        # Define the generator network
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
            nn.Softmax(dim=1)  # Ensure the quantities sum to 1
        )

    def forward(self, batch_size):
        # Generate latent vectors
        latent_vectors = torch.randn(batch_size, self.latent_dim)
        # Transform latent vectors into quantities
        quantities = self.model(latent_vectors)
        return quantities
"""

from classes import ElementGenerator
from classes import ElementDiscriminator

# Load the real alloy dataset (gan_alloy_datasets)
gan_alloy_df = pd.read_csv(project_path+"\\data\\refinedData\\gan_alloy_dataset.csv")

# Preprocess the real data: Combine elements, quantities, and features
"""
def preprocess_real_data(df):
    real_data = []
    for index, row in df.iterrows():
        element_features = []
        num_elements = 0
        for i in range(1, 10):  # For each of the 9 possible elements
            element_col = f"Element{i}"
            quantity_col = f"Quantity{i}"
            
            if pd.notna(row[element_col]):  # Only process valid elements
                element = row[element_col]
                quantity = row[quantity_col]
                
                # Get the element's properties (7 features)
                element_data = element_properties.get(element, None)
                if element_data:
                    # Add the 7 features to the element_features list
                    element_features.extend([value for value in element_data.values()])
                    
                    # Add the quantity for this element
                    element_features.append(quantity)
                    
                    # Count the valid element
                    num_elements += 1
        
        # Ensure the feature vector has the correct length (8 for each element: 7 features + 1 quantity)
        if len(element_features) != num_elements * 8:
            print(f"Error: Length mismatch for row {index}. Expected {num_elements * 8}, got {len(element_features)}.")
        
        real_data.append(element_features)
    
    return torch.tensor(real_data, dtype=torch.float32)
"""

import numpy as np
def preprocess_data(data):
    features = []
    
    # Iterate over each row in the dataset (each alloy)
    for _, row in data.iterrows():
        alloy_features = []
        
        # For each element (Element1 to Element9), extract the properties
        for i in range(1, 10):  # Loop through elements 1 to 9
            element_prefix = f"Element{i}"
            quantity_prefix = f"Quantity{i}"
            atomic_mass_prefix = f"AtomicMass{i}"
            electronegativity_prefix = f"Electronegativity{i}"
            valence_prefix = f"Valence{i}"
            electron_affinity_prefix = f"ElectronAffinity{i}"
            crystal_structure_prefix = f"Crystal Structure{i}"
            melting_point_prefix = f"Absolute Melting Point{i}"
            
            # Extract element properties
            element = row[element_prefix]
            quantity = row[quantity_prefix]
            atomic_mass = row[atomic_mass_prefix] if not np.isnan(row[atomic_mass_prefix]) else 0.0
            electronegativity = row[electronegativity_prefix] if not np.isnan(row[electronegativity_prefix]) else 0.0
            valence = row[valence_prefix] if not np.isnan(row[valence_prefix]) else 0.0
            electron_affinity = row[electron_affinity_prefix] if not np.isnan(row[electron_affinity_prefix]) else 0.0
            
            # Handle crystal structure (assuming it's a string or categorical value)
            crystal_structure = row[crystal_structure_prefix]
            if isinstance(crystal_structure, str):  # If it's a string, convert to a one-hot encoding
                crystal_structure = [1 if x == crystal_structure else 0 for x in unique_structures]
            else:
                crystal_structure = [0] * 14  # Default to a zero vector if it's not valid
            
            melting_point = row[melting_point_prefix] if not np.isnan(row[melting_point_prefix]) else 0.0
            
            # Flatten the features for this element
            element_features = [
                atomic_mass,
                electronegativity,
                valence,
                electron_affinity,
                *crystal_structure,  # Unpack the crystal structure list
                melting_point
            ]
            
            # Add the element's features to the alloy's feature list
            alloy_features.extend(element_features)
        
        # Add the alloy features to the main list
        features.append(alloy_features)
    
    # Convert the list of features into a numpy array and then a tensor
    features_array = np.array(features)
    return torch.tensor(features_array, dtype=torch.float32)





"""
# Preprocess real data from gan_alloy_datasets
real_data = preprocess_real_data(gan_alloy_df)

latent_dim = 128  # Dimension of the latent space (adjust as needed)
feature_dim = 54  # Example: 9 elements with 5 properties each + 9 quantities

# Initialize the generator
generator = ElementGenerator(latent_dim=latent_dim, feature_dim=feature_dim)
discriminator = ElementDiscriminator()

# Loss and optimizer
criterion = nn.BCELoss()  # Binary Cross Entropy loss
optimizer_g = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))


# Training loop
epochs = 50
for epoch in range(epochs):
    # Generate fake data (elements and quantities) using the generator
    elements, quantities = generator(batch_size=1)
    print("Elements:\b",elements)
    print("Quantities:\n",quantities)

    # Get the features of the generated elements
    element_properties_generated = get_element_properties(elements)
    num_elements = len(element_properties_generated)
    #quantities = quantities[:, :num_elements]
    #quantities = x[:, 72:72 + num_elements]

    print(element_properties_generated)

    # Ensure flattened_features is a single list of numbers
    flattened_features = []
    for prop in element_properties_generated:
        for value in prop.values():
            if isinstance(value, list):  # Handle one-hot encoded lists
                flattened_features.extend(value)
            else:
                flattened_features.append(value)

    # Convert quantities to a flat list of numbers
    quantities_list = quantities.view(-1).tolist()  # Flatten quantities tensor

    # Combine the flattened features with the quantities
    input_data = flattened_features + quantities_list


    print("Flattened Features:", flattened_features)
    print("Quantities List:", quantities_list)
    print("Quantities Tensor:", quantities)
    print("Quantities Shape:", quantities.shape)
    print("Input Data:", input_data)  # Debugging: Check combined data

    # Convert input data to a tensor
    input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

    # Train the discriminator: Real data (1) vs Fake data (0)
    optimizer_d.zero_grad()

    # Real data: Sample a random real data point
    real_sample = real_data[random.randint(0, len(real_data) - 1)].unsqueeze(0)

    # Labels for real and fake data
    real_labels = torch.ones(1, 1)  # Labels for real data (1 for real)
    fake_labels = torch.zeros(1, 1)  # Labels for fake data (0 for fake)

    # Compute discriminator loss on real data
    real_output = discriminator(real_sample)
    real_loss = criterion(real_output, real_labels)

    # Compute discriminator loss on fake data
    fake_output = discriminator(input_tensor)
    fake_loss = criterion(fake_output, fake_labels)

    # Total discriminator loss
    d_loss = real_loss + fake_loss

    # Backpropagate discriminator loss
    d_loss.backward()
    optimizer_d.step()

    # Train the generator: We want the generator to fool the discriminator into thinking the fake data is real
    optimizer_g.zero_grad()

    # Generator loss (we want the discriminator to classify generated data as real)
    generator_loss = criterion(fake_output, real_labels)  # We use real labels because we want the generator to fool the discriminator

    # Backpropagate generator loss
    generator_loss.backward()
    optimizer_g.step()

    # Print the losses at intervals
    #if epoch % 100 == 0:
    print(f"Epoch [{epoch}/{epochs}], D Loss: {d_loss.item()}, G Loss: {generator_loss.item()}")
"""



# Training loop
epochs = 50  # Number of epochs
batch_size = 64  # Batch size
# Preprocess real data from gan_alloy_datasets
real_data = preprocess_data(gan_alloy_df)  # Ensure real data is in the same format as generated data

feature_dim = 54  # Number of features (adjust based on dataset)
latent_dim = 128  # Dimension of the latent space
max_elements = 9
num_elements = 118
batch_size = 32
epochs = 50
learning_rate = 0.0002

# Initialize models
generator = ElementGenerator(latent_dim=latent_dim, max_elements=max_elements, num_elements=num_elements)
discriminator = ElementDiscriminator(max_elements=max_elements, num_elements=num_elements)

# Loss function and optimizers
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer_g = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

for epoch in range(epochs):
    ############################
    # Train the Discriminator
    ############################
    optimizer_d.zero_grad()

    """
    # Sample real data
    real_sample = real_data[random.randint(0, len(real_data) - 1)].unsqueeze(0)  # Randomly pick a real sample
    real_labels = torch.ones((1, 1))  # Label for real data is 1

    # Pass real data through discriminator
    #real_output = discriminator(real_sample)

    real_element_indices, real_quantities = real_sample  # Split real sample into indices and quantities

    # For fake data, you should generate the element indices and quantities using the generator
    fake_element_indices, fake_quantities = generator(batch_size)

    # Now, pass both element indices and quantities to the discriminator
    real_output = discriminator(real_element_indices, real_quantities)
    fake_output = discriminator(fake_element_indices, fake_quantities)
    """

        # Assuming real_sample is a flattened tensor with 9 elements and their corresponding properties
    real_sample = real_data[random.randint(0, len(real_data) - 1)].unsqueeze(0)  # Randomly pick a real sample
    

        # Unpack real sample into element indices and quantities
    real_element_indices = real_sample[:, :9]  # First 9 values are element indices
    real_quantities = real_sample[:, 9:18]  # Next 9 values are quantities

    # Clip element indices to ensure they are within the valid range
    real_element_indices = torch.clamp(real_element_indices, 0, num_elements - 1)

    # Create label for real data (1)
    real_labels = torch.ones((1, 1))  # Label for real data is 1

    # For fake data, generate the element indices and quantities using the generator
    fake_element_indices, fake_quantities = generator(batch_size)

    # Clip fake element indices to ensure they are within the valid range
    fake_element_indices = torch.clamp(fake_element_indices, 0, num_elements - 1)

    # Pass both element indices and quantities to the discriminator
    real_output = discriminator(real_element_indices, real_quantities)
    fake_output = discriminator(fake_element_indices, fake_quantities)

    # Calculate loss for real data
    real_loss = criterion(real_output, real_labels)



    #real_loss = criterion(real_output, real_labels)


    #fake_labels = torch.zeros((1, 1))  # Label for fake data is 0
    fake_labels = torch.zeros((1, 1), dtype=torch.float32)

    # Generate fake data
    fake_elements, fake_quantities = generator(batch_size=1)

    fake_properties = []
    for element in fake_elements:
        # Iterate over each element in the tensor
        for e in element:
            if e.item() in element_properties:
                properties = list(element_properties[e.item()].values())  # Extract values as a list of numbers
                # Flatten any sublist inside properties (if any)
                flat_properties = [item for sublist in properties for item in (sublist if isinstance(sublist, list) else [sublist])]
                fake_properties.append(flat_properties)
            else:
                print(f"Warning: Element {e.item()} not found in dataset.")
                fake_properties.append([0] * feature_dim)  # Use a default zero vector for missing properties


    #print(fake_properties)

    # Flatten properties and combine with quantities
    fake_properties_flat = torch.tensor([item for sublist in fake_properties for item in sublist], dtype=torch.float32)

    # Ensure the dimensions are correct (flatten quantities as well)
    fake_input = torch.cat((fake_properties_flat, fake_quantities.flatten()), dim=0).unsqueeze(0)



    # Ensure fake_input has the same size as expected by the discriminator (e.g., 171 features)
    expected_size = 171
    fake_input_size = fake_properties_flat.size(0) + fake_quantities.flatten().size(0)

    # If fake_input_size is smaller than expected, pad with zeros
    if fake_input_size < expected_size:
        padding = expected_size - fake_input_size
        fake_input = torch.cat((fake_properties_flat, fake_quantities.flatten()), dim=0).unsqueeze(0)
        fake_input = torch.cat((fake_input, torch.zeros(1, padding)), dim=1)  # Pad with zeros
    else:
        fake_input = torch.cat((fake_properties_flat, fake_quantities.flatten()), dim=0).unsqueeze(0)


    """
    #fake_output = discriminator(fake_input)

    # Apply sigmoid to ensure output is between 0 and 1
    #fake_output = torch.sigmoid(fake_output)
    fake_output = torch.sigmoid(fake_output.clamp(min=-10, max=10))  # Clamps values to a stable range
    #print(fake_output)
    # Check if any value in fake_output is NaN
    if torch.isnan(fake_output).any():
        #print("NaN detected in fake_output!")
        
        # Set loss to 0.99, ensure fake_output has the same shape as real_labels, and requires gradients
        fake_loss = torch.tensor(0.99, requires_grad=True).to(fake_output.device).view(-1, 1)  # Ensure requires_grad=True
        fake_output = fake_loss  # Set fake_output to the same value
    else:
        fake_loss = criterion(fake_output, fake_labels)  # Normal loss calculation if no NaN

    
    #fake_loss = criterion(fake_output, fake_labels)
    """


    fake_output = torch.sigmoid(fake_output.clamp(min=-10, max=10))  # Clamps values to a stable range

    # Ensure fake_labels has the same shape as fake_output
    fake_labels = torch.ones((fake_output.size(0), 1), device=fake_output.device)  # Adjust the shape of fake_labels

    # Check if any value in fake_output is NaN
    if torch.isnan(fake_output).any():
        # Set loss to 0.99 if NaN is detected
        fake_loss = torch.tensor(0.99, requires_grad=True).to(fake_output.device).view(-1, 1)  # Ensure requires_grad=True
        fake_output = fake_loss  # Set fake_output to the same value
    else:
        fake_loss = criterion(fake_output, fake_labels)  # Normal loss calculation if no NaN



    """
    # Generator loss: we want the discriminator to classify fake data as real
    g_loss = criterion(fake_output, real_labels)

    # Backpropagation and optimization for generator
    optimizer_g.zero_grad()  # Zero gradients for generator
    #g_loss.backward()  # Backpropagate generator loss
    optimizer_g.step()  # Update generator

    # Discriminator loss: sum of real and fake losses
    d_loss = real_loss + fake_loss

    # Backpropagation and optimization for discriminator
    optimizer_d.zero_grad()  # Zero gradients for discriminator
    d_loss.backward(retain_graph=True)  # Backpropagate discriminator loss (no retain_graph)

    g_loss.backward()  # Backpropagate generator loss


    optimizer_d.step()  # Update discriminator
    """


    real_labels = torch.ones(fake_output.size(0), 1, device=fake_output.device)  # Shape: [32, 1]
    fake_labels = torch.zeros(fake_output.size(0), 1, device=fake_output.device)  # Shape: [32, 1]



    # Generator loss: we want the discriminator to classify fake data as real
    # Generator loss: we want the discriminator to classify fake data as real
    #real_labels = torch.ones(fake_output.size(0), 1, device=fake_output.device)  # Ensure real_labels has the same size as fake_output
    g_loss = criterion(fake_output, real_labels)

    # Backpropagation and optimization for generator
    

    # Discriminator loss: sum of real and fake losses
    d_loss = real_loss + fake_loss

    # Backpropagation and optimization for discriminator
    optimizer_d.zero_grad()  # Zero gradients for discriminator
    d_loss.backward(retain_graph=True)  # Backpropagate discriminator loss (retain_graph=True for later use)
    optimizer_d.step()  # Update discriminator


    optimizer_g.zero_grad()  # Zero gradients for generator
    g_loss.backward()  # Backpropagate generator loss
    optimizer_g.step()  # Update generator


    print(f"Epoch [{epoch}/{epochs}], D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

print("Training completed.")

# You can save the model after training if needed
torch.save(generator.state_dict(), 'alloy_generator.pth')
torch.save(discriminator.state_dict(), 'alloy_discriminator.pth')
