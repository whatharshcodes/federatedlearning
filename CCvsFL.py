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