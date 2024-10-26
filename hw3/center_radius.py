import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from interval_model import Interval, IntervalReLU
from model_utils import Normalize

# Center-Radius representation for models, instead of interval rep
class CenterRadius:
    def __init__(self, center, radius):
        self.center = center  # mu
        self.radius = radius  # r
        
    @classmethod
    def from_interval(cls, lower, upper):
        center = (lower + upper) / 2
        radius = (upper - lower) / 2
        return cls(center, radius)
    
    @classmethod
    def from_interval(cls, interval: Interval):
        return cls.from_interval(interval.lower, interval.upper)
    
    def to_interval(self):
        return Interval(self.center - self.radius, self.center + self.radius)
    
    def __add__(self, other):
        if isinstance(other, CenterRadius):
            return CenterRadius(
                self.center + other.center,
                self.radius + other.radius
            )
        elif isinstance(other, (int, float)):
            return CenterRadius(
                self.center + other,
                self.radius
            )
        else:
            raise NotImplementedError("not implemented")
    
    def __sub__(self, other):
        if isinstance(other, CenterRadius):
            return CenterRadius(
                self.center - other.center,
                self.radius + other.radius
            )
        elif isinstance(other, (int, float)):
            return CenterRadius(
                self.center - other,
                self.radius
            )
        else:
            raise NotImplementedError("not implemented")
    
    def __mul__(self, other):
        if isinstance(other, (int, float)):
            center = self.center * other
            radius = abs(other) * self.radius
            return CenterRadius(center, radius)
        else:
            raise NotImplementedError("not implemented")

class CenterRadiusLinear(nn.Module):
    def __init__(self, layer: nn.Linear):
        super(CenterRadiusLinear, self).__init__()
        self.W = layer.weight.t()
        self.b = layer.bias
        
    def forward(self, x: CenterRadius) -> CenterRadius:
        mu_k = torch.matmul(self.W, x.center) + self.b
        r_k = torch.matmul(torch.abs(self.W), x.radius)
        return CenterRadius(mu_k, r_k)

    def __del__(self):
        if hasattr(self, 'W') and self.W.device != 'cpu':
            self.W.data = self.W.data.cpu()
            self.b.data = self.b.data.cpu()
        if hasattr(self, 'W'):
            del self.W
        if hasattr(self, 'b'):
            del self.b

class CenterRadiusReLU(nn.Module):
    def __init__(self):
        super(CenterRadiusReLU, self).__init__()
    
    def forward(self, x: CenterRadius) -> CenterRadius:
        # Convert to interval bounds for ReLU
        interval = x.to_interval()
        interval_relu = IntervalReLU()(interval)
        return CenterRadius.from_interval(interval_relu)

class CenterRadiusNormalize(nn.Module):
    def forward(self, x: CenterRadius) -> CenterRadius:
        return CenterRadius(
            (x.center - 0.1307) * (1/0.3081),
            x.radius * (1/0.3081)
        )

class CenterRadiusModel(nn.Module):
    def __init__(self, model: nn.Sequential):
        super(CenterRadiusModel, self).__init__()
        self.model = nn.ModuleList(self.convert2modulelist(model))
        
    def convert2modulelist(self, model: nn.Module):
        layers = []
        for layer in model.children():
            if isinstance(layer, nn.Sequential):
                layers.extend(self.convert2modulelist(layer))
            elif isinstance(layer, nn.Linear):
                layers.append(CenterRadiusLinear(layer))
            elif isinstance(layer, nn.ReLU):
                layers.append(CenterRadiusReLU())
            elif isinstance(layer, Normalize):
                layers.append(CenterRadiusNormalize())
            elif isinstance(layer, nn.Module):
                layers.extend(self.convert2modulelist(layer))
            else:
                raise NotImplementedError(f"Layer type {type(layer)} not implemented")
        return layers
    
    def forward(self, x: CenterRadius):
        assert x.center.shape == x.radius.shape
        if x.center.dim() == 4:
            b, c, h, w = x.center.size()
            x.center = x.center.view(-1, c*h*w)
            x.radius = x.radius.view(-1, c*h*w)
        
        for layer in self.model:
            x = layer(x)
        return x

    def __del__(self):
        if hasattr(self, 'model'):
            for layer in self.model:
                del layer
            del self.model