import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import copy

# Define a simple CNN for MNIST classification
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        # Apply convolution, relu and pooling sequentially
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        # Flatten the output tensor for the fully connected layers
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# Function to partition the dataset among clients
def split_dataset(dataset, num_clients):
    data_size = len(dataset)
    indices = np.arange(data_size)
    np.random.shuffle(indices)
    # Split indices equally among clients
    client_indices = np.array_split(indices, num_clients)
    return client_indices

# Perform local training for one client
def local_train(model, train_loader, epochs, device):
    model.train()  # Set model to training mode
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.NLLLoss()
    # Loop over local epochs
    for epoch in range(epochs):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()      # Reset gradients for this batch
            output = model(data)       # Forward pass
            loss = criterion(output, target)
            loss.backward()            # Backward pass
            optimizer.step()           # Update model parameters
    return model.state_dict()

# Aggregate client models using Federated Averaging
def federated_avg(state_dicts):
    avg_state_dict = copy.deepcopy(state_dicts[0])
    # Sum the corresponding model parameters from each client
    for key in avg_state_dict.keys():
        for i in range(1, len(state_dicts)):
            avg_state_dict[key] += state_dicts[i][key]
        # Average the parameters
        avg_state_dict[key] = torch.div(avg_state_dict[key], len(state_dicts))
    return avg_state_dict

# Evaluate the global model on the test dataset
def evaluate(model, test_loader, device):
    model.eval()  # Set the model to evaluation mode
    test_loss = 0
    correct = 0
    criterion = nn.NLLLoss(reduction='sum')
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # Sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)        # Get predictions
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')
    return test_loss, accuracy

def main():
    # Federated learning settings
    num_clients = 5        # Number of simulated clients
    local_epochs = 2       # Local training epochs per client
    global_rounds = 10     # Number of global aggregation rounds
    batch_size = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data transformations and loading MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('../data', train=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Partition the training dataset among the clients
    client_indices = split_dataset(train_dataset, num_clients)
    client_loaders = []
    for indices in client_indices:
        subset = Subset(train_dataset, indices)
        loader = DataLoader(subset, batch_size=batch_size, shuffle=True)
        client_loaders.append(loader)

    # Initialize the global model
    global_model = CNNModel().to(device)
    global_model.train()

    # Federated learning process: local training followed by global aggregation
    for round in range(global_rounds):
        print(f'Global Round {round+1}/{global_rounds}')
        local_state_dicts = []
        # Each client performs local training
        for client_idx, loader in enumerate(client_loaders):
            local_model = copy.deepcopy(global_model)
            local_state = local_train(local_model, loader, local_epochs, device)
            local_state_dicts.append(local_state)
            print(f'  Client {client_idx+1} finished local training.')
        
        # Aggregate client models to update the global model
        global_state_dict = federated_avg(local_state_dicts)
        global_model.load_state_dict(global_state_dict)

        # Evaluate the updated global model on the test set
        evaluate(global_model, test_loader, device)

if __name__ == "__main__":
    main()
