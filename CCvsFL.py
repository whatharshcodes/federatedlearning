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