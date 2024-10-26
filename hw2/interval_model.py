import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model_utils import Normalize


class Interval:
    #interval class with +, -, *
    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper
        
    def __add__(self, other):
        if isinstance(other, Interval):
            return Interval(self.lower + other.lower, self.upper + other.upper)
        elif isinstance(other, (int, float)):
            return Interval(self.lower + other, self.upper + other)
        else:
            raise NotImplementedError("not implemented")
    
    def __sub__(self, other):
        if isinstance(other, Interval):
            return Interval(self.lower - other.upper, self.upper - other.lower)
        elif isinstance(other, (int, float)):
            return Interval(self.lower - other, self.upper - other)
        else:
            raise NotImplementedError("not implemented")
    
    def __mul__(self, other):
        if isinstance(other, Interval):
            #not tested, mostly will throw error
            print("Warning: Interval multiplication with Interval is not tested")
            return Interval(
            min(self.lower*other.lower, self.lower*other.upper, self.upper*other.lower, self.upper*other.upper),
            max(self.lower*other.lower, self.lower*other.upper, self.upper*other.lower, self.upper*other.upper)
        )
        elif isinstance(other, (int, float)):
            if other < 0:
                return Interval(self.upper*other, self.lower*other)
            else:
                return Interval(self.lower*other, self.upper*other)
        else:
            raise NotImplementedError("not implemented")
    
            

def interval_max(x: Interval, y: Interval):
    return Interval(torch.max(x.lower, y.lower), torch.max(x.upper, y.upper))

class IntervalLinear(nn.Module):
    def __init__(self, layer: nn.Linear):
        super(IntervalLinear, self).__init__()
        self.W = layer.weight.t()
        self.b = layer.bias
        
    def forward(self, x: Interval) -> Interval:
        W_pos = torch.clamp(self.W, min=0)
        W_neg = torch.clamp(self.W, max=0)
        x_l = x.lower
        x_u = x.upper
        
        final_lower = torch.matmul(x_l, W_pos) + torch.matmul(x_u, W_neg) + self.b
        final_upper = torch.matmul(x_u, W_pos) + torch.matmul(x_l, W_neg) + self.b
        
        return Interval(final_lower, final_upper)

    def __delitem__(self):
        if self.W.device != 'cpu':
            self.W.data = self.W.data.cpu()
            self.b.data = self.b.data.cpu()
        del self.W
        del self.b
        
class IntervalReLU(nn.Module):
    def __init__(self):
        super(IntervalReLU, self).__init__()
    
    def forward(self, x: Interval) -> Interval:
        return Interval(F.relu(x.lower), F.relu(x.upper))
    
class IntervalModel(nn.Module):
    def __init__(self, model: nn.Sequential):
        #We convert sequential model to interval model by overriding the forward function
        super(IntervalModel, self).__init__()
        self.model = nn.ModuleList(self.convert2modulelist(model))
    def convert2modulelist(self, model: nn.Module):
        layers = []
        for layer in model.children():
            if isinstance(layer, nn.Sequential):
                layers.extend(self.convert2modulelist(layer))
            elif isinstance(layer, nn.Linear):
                layers.append(IntervalLinear(layer))
            elif isinstance(layer, nn.ReLU):
                layers.append(IntervalReLU())
            elif isinstance(layer, Normalize):
                layers.append(Normalize())
            elif isinstance(layer, nn.Module):
                layers.extend(self.convert2modulelist(layer))
            else:
                raise NotImplementedError("Not implemented")
        return layers
    
    def forward(self, x):
        assert x.lower.shape == x.upper.shape
        if x.lower.dim() == 4:
            b, c, h, w = x.lower.size()
            x.lower = x.lower.view(-1, c*h*w) #(B, C, H, W) -> (B, C*H*W)
            x.upper = x.upper.view(-1, c*h*w)
        for layer in self.model:
            x = layer(x)
        return x

    def __delitem__(self):
        for layer in self.model:
            del layer
        del self.model