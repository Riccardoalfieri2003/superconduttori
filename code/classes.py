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
    def __init__(self, latent_dim, max_elements=9, num_elements=118):
        super(ElementGenerator, self).__init__()
        self.latent_dim = latent_dim
        self.max_elements = max_elements  # Maximum number of elements in an alloy
        self.num_elements = num_elements  # Total number of elements (118 for the periodic table)

        # Define the generator network for quantities
        self.quantity_model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, max_elements)  # Output: quantities for up to 9 elements
        )

        # Define a linear layer for element indices
        self.element_model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_elements),  # Output: probabilities for all 118 elements
            nn.Softmax(dim=1)  # Normalize probabilities
        )

    def forward(self, batch_size):
        # Generate latent vectors
        latent_vectors = torch.randn(batch_size, self.latent_dim)

        # Generate quantities
        quantities = self.quantity_model(latent_vectors)

        # Generate element indices
        element_probs = self.element_model(latent_vectors)  # Probabilities for 118 elements
        element_indices = torch.multinomial(element_probs, self.max_elements, replacement=False)

        return element_indices, quantities






class ElementDiscriminator(nn.Module):
    def __init__(self, max_elements=9, num_elements=118):
        super(ElementDiscriminator, self).__init__()
        self.max_elements = max_elements  # Store max_elements
        self.num_elements = num_elements  # Store num_elements
        self.input_dim = max_elements * (1 + num_elements)  # Each element has a one-hot vector + quantity

        self.model = nn.Sequential(
            nn.Linear(self.input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),  # Output: real or fake
            nn.Sigmoid()  # Sigmoid for binary classification
        )

    def forward(self, element_indices, quantities):
        # Ensure element_indices are of type int64
        element_indices = element_indices.to(torch.int64)

        # One-hot encode element indices
        batch_size, max_elements = element_indices.size()
        one_hot = torch.zeros(batch_size, max_elements, self.num_elements, device=element_indices.device)
        one_hot.scatter_(2, element_indices.unsqueeze(-1), 1)

        # Flatten one-hot and concatenate with quantities
        one_hot_flat = one_hot.view(batch_size, -1)
        input_data = torch.cat([one_hot_flat, quantities], dim=1)

        return self.model(input_data)

