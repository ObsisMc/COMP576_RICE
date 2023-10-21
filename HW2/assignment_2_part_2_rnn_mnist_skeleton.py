# -*- coding: utf-8 -*-
"""Assignment_2_Part_2_RNN_MNIST_vp1.ipynb
Overall structure:

1) Set Pytorch metada
- seed
- tensorflow output
- whether to transfer to gpu (cuda)

2) Import data
- download data
- create data loaders with batchsie, transforms, scaling

3) Define Model architecture, loss and optimizer

4) Define Test and Training loop
    - Train:
        a. get next batch
        b. forward pass through model
        c. calculate loss
        d. backward pass from loss (calculates the gradient for each parameter)
        e. optimizer: performs weight updates

5) Perform Training over multiple epochs:
    Each epoch:
    - call train loop
    - call test loop

# Step 1: Pytorch and Training Metadata
"""

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
import matplotlib.pyplot as plt

batch_size = 64
test_batch_size = 1000
epochs = 10
lr = 0.01
try_cuda = True
seed = 1000
logging_interval = 10  # how many batches to wait before logging
logging_dir = None

INPUT_SIZE = 28
HIDDEN_SIZE = 128
NUM_LAYERS = 2

# 1) setting up the logging
model_name = f"lstm_lr{lr}_epoch{epochs}_hs{HIDDEN_SIZE}_nl{NUM_LAYERS}_adam"
datetime_str = datetime.now().strftime('%b%d_%H-%M-%S')

if logging_dir is None:
    runs_dir = Path("./") / Path(f"runs/") / Path(f"part2")
    runs_dir.mkdir(exist_ok=True)

    logging_dir = runs_dir / Path(f"{model_name}_{datetime_str}")

    logging_dir.mkdir(exist_ok=True)
    logging_dir = str(logging_dir.absolute())

writer = SummaryWriter(log_dir=logging_dir)

# deciding whether to send to the cpu or not if available
if torch.cuda.is_available() and try_cuda:
    cuda = True
    device = "cuda"
    torch.cuda.manual_seed(seed)
else:
    cuda = False
    device = "cpu"
    torch.manual_seed(seed)

"""# Step 2: Data Setup"""

# Setting up data
transform = transforms.Compose([
    transforms.ToTensor()
])

train_dataset = datasets.MNIST(
    './data',
    train=True,
    download=True,
    transform=transform
)
test_dataset = datasets.MNIST(
    './data',
    train=False,
    download=True,
    transform=transform
)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

# plot one example
print(train_dataset.train_data.size())  # (60000, 28, 28)
print(train_dataset.train_labels.size())  # (60000)
plt.imshow(train_dataset.train_data[0].numpy(), cmap='gray')
plt.title('%i' % train_dataset.train_labels[0])
plt.show()

"""# Step 3: Creating the Model"""


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.num_layers = NUM_LAYERS
        self.hidden_size = HIDDEN_SIZE
        # self.rnn = nn.RNN(INPUT_SIZE, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True,
        #                   nonlinearity='relu')
        self.rnn = nn.LSTM(INPUT_SIZE, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
        # self.rnn = nn.GRU(INPUT_SIZE, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
        self.out = nn.Linear(self.hidden_size, 10)

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        # c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        if cuda:
            h0 = h0.cuda()
        #     c0 = c0.cuda()

        r_out, hidden = self.rnn(x, None)  # None represents zero initial hidden state

        # choose r_out at the last time step
        out = self.out(r_out[:, -1, :])
        return out


model = Net()

if cuda:
    model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

"""# Step 4: Train/Test"""


# Defining the test and trainig loops

def train(epoch):
    model.train()

    criterion = nn.CrossEntropyLoss()
    loss_all = 0
    accu_all = 0
    print(f"Train loss  Accuracy")
    for batch_idx, (data, target) in enumerate(train_loader):
        if cuda:
            data, target = data.cuda(), target.cuda()

        data = data.view(-1, 28, 28)

        optimizer.zero_grad()
        output = model(data)  # forward
        loss = criterion(output, target)

        loss.backward()
        optimizer.step()

        loss_all += loss.item() / len(train_loader)
        accu = (output.argmax(dim=-1) == target).sum() / len(target)
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
        if cuda:
            data, target = data.cuda(), target.cuda()

        data = data.view(-1, 28, 28)
        output = model(data)  # forward
        loss = criterion(output, target)

        test_loss += loss.item() / len(test_loader)

        accu = (output.argmax(dim=-1) == target).sum() / len(target)
        correct += accu / len(test_loader)
        print(f"Test{epoch}: {loss:.5f}    {accu:.3f}")

    writer.add_scalar("test/loss", test_loss, epoch)
    writer.add_scalar("test/accuracy", correct, epoch)
    print(f"---> Epoch Test Loss: {test_loss}, accuracy: {correct}")

    if test_loss < best_loss:
        print("Save model...")
        best_loss = test_loss
        torch.save(model.state_dict(), f"./part2/{model_name}")


# Training loop

if __name__ == "__main__":
    if not os.path.exists("./part2"):
        os.mkdir("./part2")
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
