import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt
from assignment_2_part_1_cifar10_skeleton import LeNet5


def visualize_weight(ckpt_path):
    state_dict = torch.load(ckpt_path)
    conv1_weight = state_dict["features.0.weight"]

    npimg = conv1_weight.cpu().numpy()
    C_out, C_in, H, W = npimg.shape
    fig = plt.figure()
    for i in range(C_out):
        ax = fig.add_subplot(int(f"33{i % 9 + 1}"))
        ax.imshow(np.transpose(npimg[i], (1, 2, 0)), cmap="gray")
        if i % 9 == 8:
            plt.show()
            fig = plt.figure()
    plt.show()


def show_activation_statistics(ckpt_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    state_dict = torch.load(ckpt_path)
    model = LeNet5(10, True).to(device)
    model.load_state_dict(state_dict)

    # add hooks to get statistics of activations
    mean1, mean2 = [], []
    std1, std2 = [], []

    def hook_relu1(m, i, o):
        mean1.append(o.mean().item())
        std1.append(o.std().item())

    def hook_relu2(m, i, o):
        mean2.append(o.mean().item())
        std2.append(o.std().item())

    relu1: nn.Module = dict(model.named_modules())["features.1"]
    relu2: nn.Module = dict(model.named_modules())["features.4"]
    relu1.register_forward_hook(hook_relu1)
    relu2.register_forward_hook(hook_relu2)

    # load dataset
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
    ])

    test_dataset = datasets.CIFAR10(root="../data", train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # run model
    model.eval()
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        logits, probas = model(data)

    mean_act1 = torch.tensor(mean1).mean()
    std_act1 = torch.tensor(std1).mean()

    mean_act2 = torch.tensor(mean2).mean()
    std_act2 = torch.tensor(std2).mean()

    print(f"Statistics:     Mean     Std\n"
          f"Activation 1    {mean_act1:<9.4f}{std_act1:<.4f}\n"
          f"Activation 2    {mean_act2:<9.4f}{std_act2:<.4f}")


if __name__ == "__main__":
    ckpt_path = "./part1/lr0.01_epoch20_adagrad.pt"
    # visualize_weight(ckpt_path)
    show_activation_statistics(ckpt_path)
