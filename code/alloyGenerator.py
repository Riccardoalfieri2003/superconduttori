import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

import sys
import os
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_path)


# Hyperparameters
noise_dim = 100  # Dimension of the noise vector
feature_dim = 72  # Number of features per alloy (adjust based on dataset)
batch_size = 64
epochs = 100
lr = 0.0002

"""
# Load and preprocess dataset
def preprocess_data(filepath):
    data = pd.read_csv(filepath)
    data.fillna(0, inplace=True)  # Replace None with 0
    # Normalize numerical columns (adjust columns as needed)
    numerical_cols = [col for col in data.columns if "Quantity" in col or "AtomicMass" in col or "Electronegativity" in col]
    data[numerical_cols] = (data[numerical_cols] - data[numerical_cols].min()) / (data[numerical_cols].max() - data[numerical_cols].min())
    return data

"""
def preprocess_data(filepath):
    data = pd.read_csv(filepath)
    data.fillna(0, inplace=True)  # Replace None or NaN with 0
    
    # Identify and encode non-numeric columns (e.g., Element, Crystal Structure)
    #categorical_cols = ["Element1", "Crystal Structure1", "Element2", "Crystal Structure2", ...]  # Add all relevant columns

    categorical_cols = []
    for i in range(1, 10):  # Assuming the range is from 1 to 9
        categorical_cols.append(f"Element{i}")
        categorical_cols.append(f"Crystal Structure{i}")

    for col in categorical_cols:
        data[col] = data[col].astype('category').cat.codes  # Convert to numeric codes
    
    # Normalize numerical columns (optional)
    #numerical_cols = [col for col in data.columns if col not in categorical_cols]
    #data[numerical_cols] = (data[numerical_cols] - data[numerical_cols].min()) / (data[numerical_cols].max() - data[numerical_cols].min())
    
    return data

# Preprocess the data
data = preprocess_data(project_path+"\\data\\refinedData\\gan_alloy_dataset.csv")
data_tensor = torch.tensor(data.values, dtype=torch.float32)



# Generator
class Generator(nn.Module):
    def __init__(self, noise_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
            nn.Sigmoid()  # Outputs normalized values
        )
    
    def forward(self, noise):
        return self.model(noise)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()  # Probability of being real
        )
    
    def forward(self, x):
        return self.model(x)

# Initialize models
generator = Generator(noise_dim, feature_dim)
discriminator = Discriminator(feature_dim)

# Optimizers
g_optimizer = optim.Adam(generator.parameters(), lr=lr)
d_optimizer = optim.Adam(discriminator.parameters(), lr=lr)

# Loss function
criterion = nn.BCELoss()

# Training loop
for epoch in range(epochs):
    for i in range(0, len(data_tensor), batch_size):
        real_data = data_tensor[i:i+batch_size]
        real_labels = torch.ones((real_data.size(0), 1))
        fake_labels = torch.zeros((real_data.size(0), 1))

        # Train Discriminator
        noise = torch.randn(real_data.size(0), noise_dim)
        fake_data = generator(noise)
        real_output = discriminator(real_data)
        fake_output = discriminator(fake_data.detach())
        d_loss_real = criterion(real_output, real_labels)
        d_loss_fake = criterion(fake_output, fake_labels)
        d_loss = d_loss_real + d_loss_fake
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        # Train Generator
        fake_output = discriminator(fake_data)
        g_loss = criterion(fake_output, real_labels)  # Trick discriminator
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

    print(f"Epoch {epoch + 1}/{epochs}, D Loss: {d_loss.item()}, G Loss: {g_loss.item()}")

# Save the models
torch.save(generator.state_dict(), "generator.pth")
torch.save(discriminator.state_dict(), "discriminator.pth")
