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

from capsule_conv_layer import CapsuleConvLayer
from capsule_layer import CapsuleLayer


class CapsuleNetwork(nn.Module):
    def __init__(self, num_conv_channels, num_primary_units, primary_unit_size, output_unit_size):
        super(CapsuleNetwork, self).__init__()

        self.conv1 = CapsuleConvLayer(in_channels=1,
                                      out_channels=num_conv_channels)

        self.primary = CapsuleLayer(in_units=0,
                                    in_channels=num_conv_channels,
                                    num_units=num_primary_units,
                                    unit_size=primary_unit_size,
                                    use_routing=False)

        self.digits = CapsuleLayer(in_units=num_primary_units,
                                   in_channels=primary_unit_size,
                                   num_units=10,
                                   unit_size=output_unit_size,
                                   use_routing=True)

    def forward(self, x):
        return self.digits(self.primary(self.conv1(x)))

    def loss(self, input, target, size_average=True):
        return self.margin_loss(input, target, size_average) + self.reconstruction_loss(input, target, size_average)

    def margin_loss(self, input, target, size_average=True):
        batch_size = input.size(0)

        # ||vc|| from the paper.
        v_mag = torch.sqrt((input**2).sum(dim=2, keepdim=True))

        # Calculate left and right max() terms from equation 4 in the paper.
        zero = Variable(torch.zeros(1)).cuda()
        m_plus = 0.9
        m_minus = 0.1
        max_l = torch.max(m_plus - v_mag, zero).view(batch_size, -1)
        max_r = torch.max(v_mag - m_minus, zero).view(batch_size, -1)

        # This is equation 4 from the paper.
        loss_lambda = 0.5
        T_c = target
        L_c = T_c * max_l + loss_lambda * (1.0 - T_c) * max_r
        L_c = L_c.sum(dim=1)

        if size_average:
            L_c = L_c.mean()

        return L_c

    def reconstruction_loss(self, input, target, size_average=True):
        # fixme implement
        return 0.0
