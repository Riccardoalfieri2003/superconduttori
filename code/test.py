import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import sys
import os
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_path)

from classes import Generator
from classes import Discriminator
# Hyperparameters
noise_dim = 100  # Dimension of the noise vector
feature_dim = 72  # Number of features per alloy (adjust based on dataset)
batch_size = 64
epochs = 100
lr = 0.0002


# Assuming you have the same dataset used for training, load it as a DataFrame
alloy_data = pd.read_csv(project_path+"\\data\\refinedData\\gan_alloy_dataset.csv")  # Replace with your actual dataset


# Load the models
generator = Generator(noise_dim, feature_dim)
discriminator = Discriminator(feature_dim)

# Load the state dictionaries into the models
generator.load_state_dict(torch.load(project_path+'\\models\\generator.pth'))
discriminator.load_state_dict(torch.load(project_path+'\\models\\discriminator.pth'))

# Ensure the models are in evaluation mode
generator.eval()
discriminator.eval()

def preprocess_elements(elements):
    # Create a list of one-hot encodings for the elements
    # Assuming you have a mapping of element names to indices in your dataset
    
    # Get all element columns (e.g., Element1, Element2, ..., Element9)
    element_columns = [f'Element{i}' for i in range(1, 10)]  # Adjust based on the number of elements in your dataset
    
    # Flatten the unique elements from all element columns
    unique_elements = pd.unique(alloy_data[element_columns].values.ravel())
    
    # Create a mapping of element names to indices
    element_mapping = {name: idx for idx, name in enumerate(unique_elements)}
    
    # Initialize an input vector with zeros
    input_vector = np.zeros(len(element_mapping))
    
    # Set 1 for the input elements in the one-hot encoded vector
    for element in elements:
        if element in element_mapping:
            input_vector[element_mapping[element]] = 1  # Set 1 for the input element
    
    # Ensure the input size matches the expected size (e.g., 100)
    expected_input_size = 100  # Adjust this based on the model's expected input size
    if len(input_vector) < expected_input_size:
        input_vector = np.pad(input_vector, (0, expected_input_size - len(input_vector)), 'constant')
    
    return torch.tensor(input_vector, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

# Function to generate an alloy and evaluate its realism
def generate_alloy(elements):
    # Preprocess the input elements
    input_data = preprocess_elements(elements)
    
    # Generate the alloy using the generator
    with torch.no_grad():  # Disable gradient calculation for inference
        generated_alloy = generator(input_data)
    
    # Evaluate the generated alloy with the discriminator
    with torch.no_grad():
        realism_score = discriminator(generated_alloy)
    
    # Print the generated alloy and its realism score
    print("Generated Alloy Composition: ", generated_alloy)
    print("Realism Score (0 = fake, 1 = real): ", realism_score.item())
    
    if realism_score.item() > 0.5:
        print("The generated alloy is realistic!")
    else:
        print("The generated alloy is not realistic.")

# Example usage:
input_elements = ["Be", "Li"]  # Elements you want to input
generate_alloy(input_elements)
