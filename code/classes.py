import torch
import torch.nn as nn


"""
# Generator
class ElementGenerator(nn.Module):
    def __init__(self, latent_dim, feature_dim, output_dim, num_elements):
        super(ElementGenerator, self).__init__()
        self.latent_dim = latent_dim
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.num_elements = num_elements  # Number of possible elements

        # Define the generator network
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
            nn.Softmax(dim=1)  # Ensure the quantities sum to 1
        )

        # Define a linear layer to generate element indices
        self.element_layer = nn.Linear(latent_dim, num_elements)

    def forward(self, batch_size):
        # Generate latent vectors
        latent_vectors = torch.randn(batch_size, self.latent_dim)

        # Generate quantities
        quantities = self.model(latent_vectors)

        # Generate element indices (e.g., integers corresponding to elements)
        element_indices = torch.argmax(self.element_layer(latent_vectors), dim=1)

        return element_indices, quantities


    

class ElementDiscriminator(nn.Module):
    def __init__(self):
        super(ElementDiscriminator, self).__init__()
        # Define layers (e.g., fully connected layers)
        #self.fc1 = nn.Linear(8 * 9, 512)  # Input size should match the generated data shape
        self.fc1 = nn.Linear(171, 512)  # Adjust input size to 171 # 171 perchè sono le feature con crystal structure encodate
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 1)  # Output: real or fake (binary)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))  # Sigmoid for binary classification
        return x
"""


class ElementGenerator(nn.Module):
    def __init__(self, latent_dim, feature_dim, num_elements):
        super(ElementGenerator, self).__init__()
        self.latent_dim = latent_dim
        self.feature_dim = feature_dim
        self.num_elements = num_elements  # Number of elements in input

        # Define the generator network
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, num_elements)  # Output matches the number of input elements
        )

    def forward(self, batch_size):
        # Generate latent vectors
        latent_vectors = torch.randn(batch_size, self.latent_dim)

        # Generate quantities (no Softmax, raw values)
        quantities = self.model(latent_vectors)

        return quantities
    


class ElementDiscriminator(nn.Module):
    def __init__(self, input_dim):
        super(ElementDiscriminator, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)  # Input size should match the number of elements or features
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 1)  # Output: real or fake (binary classification)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))  # Sigmoid for binary classification
        return x