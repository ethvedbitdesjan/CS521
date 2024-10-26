import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

## Simple NN. You can change this if you want. If you change it, mention the architectural details in your report.
class Net(nn.Module):
    def __init__(self, hidden_sizes=(50, 50, 50), num_classes=10, input_size=28*28):
        assert len(hidden_sizes) == 3
        #TODO: Implement the module list
        super(Net, self).__init__()
        self.fc = nn.Linear(input_size, hidden_sizes[0])
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_sizes[0],hidden_sizes[1])
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(hidden_sizes[2], num_classes)
        

    def forward(self, x):
        x = x.view((-1, self.fc.in_features))
        x = self.relu1(self.fc(x))
        x = self.relu2(self.fc2(x))
        # x = self.relu3(self.fc3(x))
        x = self.fc4(x)
        return x

class Normalize(nn.Module):
    def forward(self, x):
        return (x - 0.1307) * (1/0.3081) #* replaces division for intervals

# Add the data normalization as a first "layer" to the network
# this allows us to search for adverserial examples to the real image, rather than
# to the normalized image

def load_model(hidden_sizes=(50, 50, 50), num_classes=10, input_size=28*28):
    model = nn.Sequential(Normalize(), Net(hidden_sizes, num_classes, input_size))
    return model