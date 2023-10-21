# -*- coding: utf-8 -*-
"""Assignment_2_Part_1_Cifar10_vp1.ipynb

Purpose: Implement image classsification nn the cifar10
dataset using a pytorch implementation of a CNN architecture (LeNet5)

Pseudocode:
1) Set Pytorch metada
- seed
- tensorboard output (logging)
- whether to transfer to gpu (cuda)

2) Import the data
- download the data
- create the pytorch datasets
    scaling
- create pytorch dataloaders
    transforms
    batch size

3) Define the model architecture, loss and optimizer

4) Define Test and Training loop
    - Train:
        a. get next batch
        b. forward pass through model
        c. calculate loss
        d. backward pass from loss (calculates the gradient for each parameter)
        e. optimizer: performs weight updates
        f. Calculate accuracy, other stats
    - Test:
        a. Calculate loss, accuracy, other stats

5) Perform Training over multiple epochs:
    Each epoch:
    - call train loop
    - call test loop




"""

# Step 1: Pytorch and Training Metadata

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
from pathlib import Path

# hyperparameters
batch_size = 128
epochs = 20
lr = 0.01
try_cuda = True
seed = 1000

# Architecture
num_classes = 10

# otherum
logging_interval = 10  # how many batches to wait before logging
logging_dir = None
grayscale = True

# 1) setting up the logging
model_name = f"lr{lr}_epoch{epochs}_adagrad"
if __name__ == "__main__":
    datetime_str = datetime.now().strftime('%b%d_%H-%M-%S')

    if logging_dir is None:
        runs_dir = Path("./") / Path(f"runs/")
        runs_dir.mkdir(exist_ok=True)

        runs_dir = runs_dir / Path(f"part1")
        runs_dir.mkdir(exist_ok=True)

        logging_dir = runs_dir / Path(f"{model_name}_{datetime_str}")

        logging_dir.mkdir(exist_ok=True)
        logging_dir = str(logging_dir.absolute())

    writer = SummaryWriter(log_dir=logging_dir)

# deciding whether to send to the cpu or not if available
if torch.cuda.is_available() and try_cuda:
    cuda = True
    torch.cuda.manual_seed(seed)
else:
    cuda = False
    torch.manual_seed(seed)

"""# Step 2: Data Setup"""

# downloading the cifar10 dataset

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
    ])

    train_dataset = datasets.CIFAR10(root="../data", train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root="../data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    def check_data_loader_dim(loader):
        # Checking the dataset
        for images, labels in loader:
            print('Image batch dimensions:', images.shape)
            print('Image label dimensions:', labels.shape)
            break


    check_data_loader_dim(train_loader)
    check_data_loader_dim(test_loader)

"""# 3) Creating the Model"""

layer_1_n_filters = 32
layer_2_n_filters = 64
fc_1_n_nodes = 1024
padding = "zeros"
kernel_size = 5
verbose = False

# calculating the side length of the final activation maps
final_length = 7

if verbose:
    print(f"final_length = {final_length}")


class LeNet5(nn.Module):

    def __init__(self, num_classes, grayscale=False):
        super(LeNet5, self).__init__()

        self.grayscale = grayscale
        self.num_classes = num_classes

        if self.grayscale:
            in_channels = 1
        else:
            in_channels = 3

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, layer_1_n_filters, kernel_size, 1, 2, padding_mode=padding),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(layer_1_n_filters, layer_2_n_filters, kernel_size, 1, 1, padding_mode=padding),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten()
        )

        self.classifier = nn.Sequential(
            nn.Linear(final_length * final_length * layer_2_n_filters * in_channels, fc_1_n_nodes),
            nn.Tanh(),
            nn.Linear(fc_1_n_nodes, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        logits = self.classifier(x)
        probas = F.softmax(logits, dim=1)
        return logits, probas


model = LeNet5(num_classes, True)

if cuda:
    model.cuda()
    device = "cuda"

optimizer = optim.Adagrad(model.parameters(), lr=lr)

"""# Step 4: Train/Test Loop"""


# Defining the test and trainig loops

def train(epoch):
    model.train()

    criterion = nn.CrossEntropyLoss()
    loss_all = 0
    accu_all = 0
    print(f"Train loss  Accuracy")
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        logits, probas = model(data)  # forward

        loss = criterion(logits, target)
        loss_all += loss.item() / len(train_loader)
        loss.backward()
        optimizer.step()

        accu = (probas.argmax(dim=-1) == target).sum() / len(target)
        accu_all += accu / len(train_loader)
        print(f"Train{epoch}: {loss:.5f}   {accu:.3f}")

    writer.add_scalar("train/loss", loss_all, epoch)
    writer.add_scalar("train/accuracy", accu_all, epoch)
    print(f"---> Epoch Train Loss: {loss_all}, accuracy: {accu_all}")


best_loss = 1e5


@torch.no_grad()
def test(epoch):
    global best_loss
    model.eval()
    test_loss = 0
    correct = 0
    criterion = nn.CrossEntropyLoss()

    print(f"Test loss  Accuracy")
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)

        logits, probas = model(data)

        loss = criterion(logits, target)
        test_loss += loss.item() / len(test_loader)

        accu = (probas.argmax(dim=-1) == target).sum() / len(target)
        correct += accu / len(test_loader)
        print(f"Test{epoch}: {loss:.5f}    {accu:.3f}")

    writer.add_scalar("test/loss", test_loss, epoch)
    writer.add_scalar("test/accuracy", correct, epoch)
    print(f"---> Epoch Test Loss: {test_loss}, accuracy: {correct}")

    if test_loss < best_loss:
        print("Save model...")
        best_loss = test_loss
        torch.save(model.state_dict(), f"./part1/{model_name}.pt")


if __name__ == "__main__":
    if not os.path.exists("./part1"):
        os.mkdir("./part1")
    for epoch in range(epochs):
        train(epoch)
        test(epoch)

    writer.close()

# Commented out IPython magic to ensure Python compatibility.
"""
#https://stackoverflow.com/questions/55970686/tensorboard-not-found-as-magic-function-in-jupyter

#seems to be working in firefox when not working in Google Chrome when running in Colab
#https://stackoverflow.com/questions/64218755/getting-error-403-in-google-colab-with-tensorboard-with-firefox


# %load_ext tensorboard
# %tensorboard --logdir [dir]

"""
