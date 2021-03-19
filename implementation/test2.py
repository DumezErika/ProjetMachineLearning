import torch
import torch.utils.data as data
from torchdyn.datasets import *
import torch.nn as nn
from torchdyn.models import *
import matplotlib.pyplot as plt
from litneuralode import *

"""
loss = nn.CrossEntropyLoss()
input = torch.randn(3,3,requires_grad=True)
target = torch.empty(3, dtype=torch.long).random_(3)
print("input : ", input)
print("target : ", target)
output = loss(input, target)
print("output avant back : ", output)
output.backward()
print("output après back : ", output)
"""