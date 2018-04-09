#
# Dynamic Routing Between Capsules
# https://arxiv.org/pdf/1710.09829.pdf
#

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.nn.functional as F


class CapsuleReconstrutionLayer(nn.Module):
    def __init__(self, num_units, unit_size,reconstruction_size):
        super(CapsuleReconstrutionLayer, self).__init__()

        self.reconstruct0 = nn.Linear(num_units*unit_size, int((reconstruction_size * 2) / 3))
        self.reconstruct1 = nn.Linear(int((reconstruction_size * 2) / 3), int((reconstruction_size * 3) / 2))
        self.reconstruct2 = nn.Linear(int((reconstruction_size * 3) / 2), reconstruction_size)
        
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        output = self.relu(self.reconstruct0(x))
        output = self.relu(self.reconstruct1(output))
        return self.sigmoid(self.reconstruct2(output))
