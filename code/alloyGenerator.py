import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

import sys
import os
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_path)

"""
# Hyperparameters
noise_dim = 100  # Dimension of the noise vector
feature_dim = 72  # Number of features per alloy (adjust based on dataset)
batch_size = 64
epochs = 100
lr = 0.0002


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
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


"""
# Preprocess elements dataset
def load_elements(filepath):
    elements = pd.read_csv(filepath)
    elements['Index'] = range(len(elements))  # Add an index for mapping
    return elements


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


# Load datasets
elements_df = load_elements(elements_dataset_path)
gan_data = preprocess_data(gan_dataset_path)
gan_data_tensor = torch.tensor(gan_data.values, dtype=torch.float32)

# Generator
class Generator(nn.Module):
    def __init__(self, noise_dim, num_elements, max_elements):
        super(Generator, self).__init__()
        self.num_elements = num_elements
        self.max_elements = max_elements

        self.element_selector = nn.Sequential(
            nn.Linear(noise_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_elements * max_elements),
            nn.Softmax(dim=-1)
        )

        self.quantity_generator = nn.Sequential(
            nn.Linear(noise_dim, 256),
            nn.ReLU(),
            nn.Linear(256, max_elements),
            nn.Sigmoid()
        )

    def forward(self, noise):
        element_probs = self.element_selector(noise).view(-1, self.max_elements, self.num_elements)
        quantities = self.quantity_generator(noise)
        return element_probs, quantities

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
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Initialize models
#generator = Generator(noise_dim, num_elements, max_elements)
#discriminator = Discriminator(max_elements + num_elements * max_elements)
#discriminator = Discriminator(input_dim=gan_data_tensor.shape[1])

generator = Generator(noise_dim, num_elements, max_elements).to(device)
discriminator = Discriminator(input_dim=gan_data_tensor.shape[1]).to(device)
#discriminator = Discriminator(input_dim=max_elements * num_elements + max_elements).to(device)
gan_data_tensor = gan_data_tensor.to(device)


print("Shape of gan_data_tensor:", gan_data_tensor.shape)
print("Shape of real_data:", gan_data.shape)
print("Expected input size of Discriminator:", discriminator.model[0].in_features)


# Optimizers
g_optimizer = optim.Adam(generator.parameters(), lr=lr)
d_optimizer = optim.Adam(discriminator.parameters(), lr=lr)

# Loss function
criterion = nn.BCELoss()

# Training loop
for epoch in range(epochs):
    for i in range(0, len(gan_data_tensor), batch_size):
        real_data = gan_data_tensor[i:i+batch_size]
        real_labels = torch.ones((real_data.size(0), 1))
        fake_labels = torch.zeros((real_data.size(0), 1))

        # Train Discriminator
        noise = torch.randn(real_data.size(0), noise_dim)
        element_probs, quantities = generator(noise)
        
        # Sample elements from probabilities
        sampled_elements = torch.multinomial(element_probs.view(-1, num_elements), 1).view(-1, max_elements)
        



        real_labels = torch.ones((real_data.size(0), 1), device=device)
        fake_labels = torch.zeros((real_data.size(0), 1), device=device)
        noise = torch.randn(real_data.size(0), noise_dim, device=device)


        # Generate fake data
        #fake_data = generator(noise)

        # Check the shapes of element_probs and quantities
        print("Shape of element_probs:", element_probs.shape)
        print("Shape of quantities:", quantities.shape)

        # Flatten element_probs and quantities
        element_probs_flat = element_probs.view(-1, 944)  # Adjust based on the size of element_probs
        quantities_flat = quantities.view(-1, 8)  # Adjust based on the size of quantities

        # Ensure quantities has the same size as element_probs for concatenation
        # One option could be to repeat quantities to match the size of element_probs
        quantities_flat_expanded = quantities_flat.repeat(1, 944 // 8)  # Repeat to match the size

        # Concatenate element_probs_flat and quantities_flat_expanded
        fake_data = torch.cat((element_probs_flat, quantities_flat_expanded), dim=1)

        # Example of reducing the size using a linear layer
        linear_layer = nn.Linear(1888, 512)
        fake_data_resized = linear_layer(fake_data)

        # Now pass the resized fake_data to the discriminator
        fake_output = discriminator(fake_data_resized.detach())

        # Check the final shape
        #print("Shape of fake_data:", fake_data.shape)

        # Pass through discriminator
        #fake_output = discriminator(fake_data.detach())






        real_output = discriminator(real_data)

        
        d_loss_real = criterion(real_output, real_labels)
        d_loss_fake = criterion(fake_output, fake_labels)
        d_loss = d_loss_real + d_loss_fake
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        # Train Generator
        fake_output = discriminator(fake_data)
        g_loss = criterion(fake_output, real_labels)
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

    print(f"Epoch {epoch + 1}/{epochs}, D Loss: {d_loss.item()}, G Loss: {g_loss.item()}")

# Save models
torch.save(generator.state_dict(), "generator.pth")
torch.save(discriminator.state_dict(), "discriminator.pth")




# Post-processing for generated alloys
def generate_alloy(generator, noise_dim, elements_df):
    noise = torch.randn(1, noise_dim)
    element_probs, quantities = generator(noise)
    sampled_elements = torch.multinomial(element_probs.view(-1, num_elements), 1).view(-1, max_elements)
    quantities = quantities.detach().numpy().flatten()
    elements = [elements_df.iloc[idx]['Name'] for idx in sampled_elements.numpy().flatten()]
    return list(zip(elements, quantities))

# Example generation
generated_alloy = generate_alloy(generator, noise_dim, elements_df)
print("Generated Alloy:", generated_alloy)
"""


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# Hyperparameters
noise_dim = 100  # Dimension of the noise vector
num_elements = 118  # Total number of elements in the periodic table
max_elements = 9  # Maximum number of elements in an alloy
batch_size = 64
epochs = 50
lr = 0.0002




# Assuming your dataset is in a CSV file
class AlloyDataset(Dataset):
    def __init__(self, file_path):
        # Load your data (replace with your actual file path)
        self.data = pd.read_csv(file_path)
        
        # Split the data into features (element properties) and labels (quantities)
        # Here, we assume the first 54 columns are features and the next 9 are quantities
        self.features = self.data.iloc[:, :54].values  # 54 features
        self.quantities = self.data.iloc[:, 54:].values  # 9 quantities
        
    def __len__(self):
        # Return the number of samples in the dataset
        return len(self.data)
    
    def __getitem__(self, idx):
        # Return a single sample (features, quantities) as a tuple
        features = torch.tensor(self.features[idx], dtype=torch.float32)
        quantities = torch.tensor(self.quantities[idx], dtype=torch.float32)
        return features, quantities

# Paths for datasets
gan_dataset_path = project_path+"\\data\\refinedData\\gan_alloy_dataset.csv"
#elements_dataset_path = project_path+"\\data\\refinedData\\elements.csv"

# Create a DataLoader to handle batching and shuffling
data_loader = DataLoader(gan_dataset_path, batch_size=64, shuffle=True)


# Define the Generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(54 + 10, 128)  # 54 properties + noise
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 9)  # Output quantities for 9 elements
    
    def forward(self, x, z):
        x = torch.cat((x, z), dim=1)  # Concatenate input properties and noise
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Define the Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(54 + 9, 256)  # 54 properties + 9 quantities
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)  # Output probability (real or fake)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))

# Initialize models and optimizers
generator = Generator()
discriminator = Discriminator()

# Optimizers
optimizer_g = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Loss function
criterion = nn.BCELoss()

# Training loop
for epoch in range(epochs):
    for real_data, quantities in data_loader:  # Assuming data_loader is your dataset
        batch_size = real_data.size(0)
        
        # Create labels
        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)

        # Train Discriminator
        optimizer_d.zero_grad()
        
        # Real data
        real_data_combined = torch.cat((real_data, quantities), dim=1)
        real_output = discriminator(real_data_combined)
        d_loss_real = criterion(real_output, real_labels)

        # Fake data
        z = torch.randn(batch_size, 10)  # Random noise
        fake_quantities = generator(real_data, z)
        fake_data_combined = torch.cat((real_data, fake_quantities), dim=1)
        fake_output = discriminator(fake_data_combined.detach())
        d_loss_fake = criterion(fake_output, fake_labels)

        # Total Discriminator loss
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizer_d.step()

        # Train Generator
        optimizer_g.zero_grad()
        
        fake_output = discriminator(fake_data_combined)
        g_loss = criterion(fake_output, real_labels)  # Want to fool the discriminator
        g_loss.backward()
        optimizer_g.step()

    print(f"Epoch [{epoch}/{epochs}], D Loss: {d_loss.item()}, G Loss: {g_loss.item()}")

# Save models
torch.save(generator.state_dict(), "generator.pth")
torch.save(discriminator.state_dict(), "discriminator.pth")