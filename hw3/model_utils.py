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
        x = self.relu3(self.fc3(x))
        x = self.fc4(x)
        return x

class Normalize(nn.Module):
    def forward(self, x):
        return (x - 0.1307) * (1/0.3081) #* replaces division for intervals

# Add the data normalization as a first "layer" to the network
# this allows us to search for adverserial examples to the real image, rather than
# to the normalized image

class Average(nn.Module):
    def __init__(self, dim=1):
        super(Average, self).__init__()
        self.avg_dim = dim
        
    def forward(self, x, pad_mask=None):
        
        if pad_mask is None:
            return x.mean(dim=self.avg_dim)
        
        x = x * pad_mask.unsqueeze(-1).float()
        denominator = pad_mask.sum(dim=self.avg_dim, keepdim=True) + 1e-8
        return x.sum(dim=self.avg_dim) / denominator
    
def load_model(hidden_sizes=(50, 50, 50), num_classes=10, input_size=28*28):
    model = nn.Sequential(Normalize(), Net(hidden_sizes, num_classes, input_size))
    return model


class BOWModel(nn.Module):
    def __init__(self, pretrained_embeddings, pad_idx):
        super().__init__()
        vocab_size, embed_dim = pretrained_embeddings.shape
        self.embed = nn.Embedding.from_pretrained(
            torch.FloatTensor(pretrained_embeddings),
            padding_idx=pad_idx,
            freeze=True
        )
        self.pad_idx = pad_idx
        
        self.g_word = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU()
        )
        
        self.average = Average(dim=1)
        
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 2)
        )
        
    def forward(self, x):
        # batch_size, seq_len
        pad_mask = (x != self.pad_idx).bool().cuda()
        embedded = self.embed(x)  # batch_size, seq_len, embed_dim
        transformed = self.g_word(embedded)  # batch_size, seq_len, embed_dim
        averaged = self.average(transformed, pad_mask)  # batch_size, embed_dim
        logits = self.classifier(averaged)  # batch_size, 2
        return logits

    def get_transformed_embedding(self, word_idx):
        embed = self.embed(word_idx)
        return self.g_word(embed)
