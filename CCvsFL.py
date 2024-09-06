from collections import OrderedDict
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from datasets.utils.logging import disable_progress_bar
from torch.utils.data import DataLoader


# Specify the device to be used for training, either CPU or GPU
DEVICE = torch.device("cpu")  # Try "cuda" to train on GPU
print(
    f"Training on {DEVICE} using PyTorch {torch.__version__} and Flower {fl.__version__}"
)
# Disable progress bar for cleaner output
disable_progress_bar()


# Define constants for the number of clients and batch size
NUM_CLIENTS = 10
BATCH_SIZE = 32

def load_datasets():
    # Instead of passing transforms to CIFAR10(..., transform=transform)
    # we will use this function to dataset.with_transform(apply_transforms)
    # The transforms object is exactly the same
    # Initialize a federated dataset using CIFAR-10 and partition the training data into NUM_CLIENTS partitions
    fds = FederatedDataset(dataset="cifar10", partitioners={"train": NUM_CLIENTS})

    def apply_transforms(batch):
        # Define transformations: convert images to tensors and normalize them
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        # Apply transformations to each image in the batch
        batch["img"] = [transform(img) for img in batch["img"]]
        return batch

    # Create lists to store data loaders for each client's training and validation data
    trainloaders = []
    valloaders = []
    for partition_id in range(NUM_CLIENTS):
        # Load the partitioned training data for the client
        partition = fds.load_partition(partition_id, "train")
        # Apply the defined transformations
        partition = partition.with_transform(apply_transforms)
        # Split the partitioned data into training (80%) and validation (20%) sets
        partition = partition.train_test_split(train_size=0.8, seed=42)
        # Create data loaders for training and validation data
        trainloaders.append(DataLoader(partition["train"], batch_size=BATCH_SIZE))
        valloaders.append(DataLoader(partition["test"], batch_size=BATCH_SIZE))

    # Load and transform the test set
    testset = fds.load_split("test").with_transform(apply_transforms)
    # Create a data loader for the test set
    testloader = DataLoader(testset, batch_size=BATCH_SIZE)

    # Return the created data loaders for training, validation, and testing
    return trainloaders, valloaders, testloader

# Call the load_datasets function and store the resulting data loaders for training, validation, and testing
trainloaders, valloaders, testloader = load_datasets()

# Get the first batch of images and labels from the first client's training data
batch = next(iter(trainloaders[0]))
images, labels = batch["img"], batch["label"]

# Reshape and convert images to a NumPy array
# Matplotlib requires images with the shape (height, width, 3)
images = images.permute(0, 2, 3, 1).numpy()

# Denormalize images (the original normalization was: (image - 0.5) / 0.5)
# This step reverses that normalization: (image * 0.5) + 0.5
images = images / 2 + 0.5

# Create a figure and a grid of subplots with 4 rows and 8 columns
fig, axs = plt.subplots(4, 8, figsize=(12, 6))

# Loop over the images and plot them on the grid
for i, ax in enumerate(axs.flat):
    # Display the i-th image on the subplot
    ax.imshow(images[i])
    # Set the title of the subplot to the label of the i-th image
    ax.set_title(trainloaders[0].dataset.features["label"].int2str([labels[i]])[0])
    # Remove the axes for a cleaner look
    ax.axis("off")

# Adjust the layout so that subplots fit into the figure area
fig.tight_layout()

# Display the plot
plt.show()


#Defining the Model

class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        # Define the first convolutional layer: 3 input channels, 6 output channels, 5x5 kernel
        self.conv1 = nn.Conv2d(3, 6, 5)
        # Define a max pooling layer with a 2x2 window and a stride of 2
        self.pool = nn.MaxPool2d(2, 2)
        # Define the second convolutional layer: 6 input channels, 16 output channels, 5x5 kernel
        self.conv2 = nn.Conv2d(6, 16, 5)
        # Define the first fully connected layer: input features (16*5*5), output features (120)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # Define the second fully connected layer: input features (120), output features (84)
        self.fc2 = nn.Linear(120, 84)
        # Define the third fully connected layer: input features (84), output features (10)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply the first convolutional layer, followed by ReLU activation and max pooling
        x = self.pool(F.relu(self.conv1(x)))
        # Apply the second convolutional layer, followed by ReLU activation and max pooling
        x = self.pool(F.relu(self.conv2(x)))
        # Flatten the tensor into a vector
        x = x.view(-1, 16 * 5 * 5)
        # Apply the first fully connected layer followed by ReLU activation
        x = F.relu(self.fc1(x))
        # Apply the second fully connected layer followed by ReLU activation
        x = F.relu(self.fc2(x))
        # Apply the third fully connected layer (output layer)
        x = self.fc3(x)
        return x
        
def train(net, trainloader, epochs: int, verbose=False):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters())
    net.train()
    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        for batch in trainloader:
            images, labels = batch["img"].to(DEVICE), batch["label"].to(DEVICE)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # Metrics
            epoch_loss += loss
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        epoch_loss /= len(trainloader.dataset)
        epoch_acc = correct / total
        if verbose:
            print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}")


def test(net, testloader):
    """Evaluate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for batch in testloader:
            images, labels = batch["img"].to(DEVICE), batch["label"].to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss /= len(testloader.dataset)
    accuracy = correct / total
    return loss, accuracy
    
    
# Initialize lists to store metrics
validation_losses = []
validation_accuracies = []

trainloader = trainloaders[0]
valloader = valloaders[0]
net = Net().to(DEVICE)

for epoch in range(40):
    train(net, trainloader, 1)
    loss, accuracy = test(net, valloader)
    print(f"Epoch {epoch+1}: validation loss {loss}, accuracy {accuracy}")

    # Save the validation loss and accuracy for plotting later
    validation_losses.append(loss)
    validation_accuracies.append(accuracy)

# Final test set performance
test_loss, test_accuracy = test(net, testloader)
print(f"Final test set performance:\n\tloss {test_loss}\n\taccuracy {test_accuracy}")

# Plotting the validation metrics
epochs = range(1, 41)  # Adjust according to the number of epochs

plt.figure(figsize=(12, 5))

# Plot Validation Loss
plt.subplot(1, 2, 1)
plt.plot(epochs, validation_losses, 'b-o', label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Validation Loss Over Epochs')
plt.legend()
plt.grid(True)

# Plot Validation Accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs, validation_accuracies, 'r-o', label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy Over Epochs')
plt.legend()
plt.grid(True)

# Show the plots
plt.tight_layout()
plt.show()

def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]
    
   class FlowerClient(fl.client.NumPyClient):
    def __init__(self, net, trainloader, valloader):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader

    def get_parameters(self, config):
        return get_parameters(self.net)

    def fit(self, parameters, config):
        set_parameters(self.net, parameters)
        train(self.net, self.trainloader, epochs=1)
        return get_parameters(self.net), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        set_parameters(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}
        
        def client_fn(cid: str) -> FlowerClient:
    """Create a Flower client representing a single organization."""

    # Load model
    net = Net().to(DEVICE)

    # Load data (CIFAR-10)
    # Note: each client gets a different trainloader/valloader, so each client
    # will train and evaluate on their own unique data
    trainloader = trainloaders[int(cid)]
    valloader = valloaders[int(cid)]

    # Create a  single Flower client representing a single organization
    return FlowerClient(net, trainloader, valloader).to_client()
    
    # Create FedAvg strategy
strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,  # Sample 100% of available clients for training
    fraction_evaluate=0.5,  # Sample 50% of available clients for evaluation
    min_fit_clients=10,  # Never sample less than 10 clients for training
    min_evaluate_clients=5,  # Never sample less than 5 clients for evaluation
    min_available_clients=10,  # Wait until all 10 clients are available
)

# Specify the resources each of your clients need. By default, each
# client will be allocated 1x CPU and 0x GPUs
client_resources = {"num_cpus": 1, "num_gpus": 0.0}
if DEVICE.type == "cuda":
    # here we are assigning an entire GPU for each client.
    client_resources = {"num_cpus": 1, "num_gpus": 1.0}

# Start simulation
fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=NUM_CLIENTS,
    config=fl.server.ServerConfig(num_rounds=20),
    strategy=strategy,
    client_resources=client_resources,
)


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}
    
# Create FedAvg strategy
strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,
    fraction_evaluate=0.5,
    min_fit_clients=10,
    min_evaluate_clients=5,
    min_available_clients=10,
    evaluate_metrics_aggregation_fn=weighted_average,  # <-- pass the metric aggregation function
)

# Start simulation
fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=NUM_CLIENTS,
    config=fl.server.ServerConfig(num_rounds=20),
    strategy=strategy,
    client_resources=client_resources,
)


# Provided data for federated learning output
rounds = list(range(1, 21))
losses = [
    0.06564855835437775, 0.055744290542602534, 0.05214762487411499, 0.04917860360145569,
    0.04748152976036071, 0.04585693078041077, 0.04529121458530426, 0.044416985082626335,
    0.043022690248489384, 0.04283880763053895, 0.042376188731193545, 0.04181219084262848,
    0.04079408128261566, 0.0405597298502922, 0.040055459880828856, 0.040156802046298984,
    0.03929776152372361, 0.03856814625263214, 0.039083913195133206, 0.03837377440929413
]
accuracies = [
    0.2572, 0.35219999999999996, 0.4002, 0.4364, 0.4593999999999999, 0.48040000000000005,
    0.484, 0.49339999999999995, 0.5134000000000001, 0.5204, 0.5226000000000001, 0.5272,
    0.5488, 0.55, 0.5494000000000001, 0.554, 0.569, 0.5791999999999999, 0.5635999999999999,
    0.5641999999999999
]

# Plotting loss
plt.figure(figsize=(12, 6))
plt.plot(rounds, losses, marker='o', linestyle='-', color='b', label='Loss')
plt.title('Federated Learning Loss Over Rounds')
plt.xlabel('Rounds')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# Plotting accuracy
plt.figure(figsize=(12, 6))
plt.plot(rounds, accuracies, marker='o', linestyle='-', color='g', label='Accuracy')
plt.title('Federated Learning Accuracy Over Rounds')
plt.xlabel('Rounds')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()
