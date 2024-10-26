# !pip install tensorboardX
import torch
import numpy as np
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
# from tensorboardX import SummaryWriter


def load_dataloader(directory='mnist_data/', batch_size=64, only_test=False):
    ## Dataloaders
    test_dataset = datasets.MNIST('mnist_data/', train=False, download=True, transform=transforms.Compose(
        [transforms.ToTensor()]
    ))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    if only_test:
        return None, test_loader
    
    train_dataset = datasets.MNIST('mnist_data/', train=True, download=True, transform=transforms.Compose(
        [transforms.ToTensor()]
    ))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader

