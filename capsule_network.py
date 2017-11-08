#
# Dynamic Routing Between Capsules
# https://arxiv.org/pdf/1710.09829.pdf
#

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.utils as vutils
import torch.nn.functional as F

from capsule_conv_layer import CapsuleConvLayer
from capsule_layer import CapsuleLayer


class CapsuleNetwork(nn.Module):
    def __init__(self,
                 image_width,
                 image_height,
                 image_channels,
                 conv_inputs,
                 conv_outputs,
                 num_primary_units,
                 primary_unit_size,
                 num_output_units,
                 output_unit_size):
        super(CapsuleNetwork, self).__init__()

        self.reconstructed_image_count = 0

        self.image_channels = image_channels
        self.image_width = image_width
        self.image_height = image_height

        self.conv1 = CapsuleConvLayer(in_channels=conv_inputs,
                                      out_channels=conv_outputs)

        self.primary = CapsuleLayer(in_units=0,
                                    in_channels=conv_outputs,
                                    num_units=num_primary_units,
                                    unit_size=primary_unit_size,
                                    use_routing=False)

        self.digits = CapsuleLayer(in_units=primary_unit_size,
                                   in_channels=num_primary_units,
                                   num_units=num_output_units,
                                   unit_size=output_unit_size,
                                   use_routing=True)

        reconstruction_size = image_width * image_height * image_channels
        # self.reconstruct0 = nn.Linear(num_output_units*output_unit_size, (reconstruction_size * 2) / 3)
        # self.reconstruct1 = nn.Linear((reconstruction_size * 2) / 3, (reconstruction_size * 3) / 2)
        self.reconstruct0 = nn.Linear(output_unit_size, 512)
        self.reconstruct1 = nn.Linear(512, 1024)
        self.reconstruct2 = nn.Linear(1024, reconstruction_size)

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.digits(self.primary(self.conv1(x)))

    def loss(self, images, input, target, size_average=True):
        return self.margin_loss(input, target, size_average) + self.reconstruction_loss(images, input, target, size_average)

    def margin_loss(self, input, target, size_average=True):
        batch_size = input.size(0)

        # ||vc|| from the paper.
        v_mag = torch.sqrt((input**2).sum(dim=2, keepdim=True))

        # Calculate left and right max() terms from equation 4 in the paper.
        zero = Variable(torch.zeros(1)).cuda()
        m_plus = 0.9
        m_minus = 0.1
        max_l = torch.max(m_plus - v_mag, zero).view(batch_size, -1) ** 2
        max_r = torch.max(v_mag - m_minus, zero).view(batch_size, -1) ** 2

        # This is equation 4 from the paper.
        loss_lambda = 0.5
        T_c = target
        L_c = T_c * max_l + loss_lambda * (1.0 - T_c) * max_r
        L_c = L_c.sum(dim=1)

        if size_average:
            L_c = L_c.mean()

        return L_c

    def reconstruction_loss(self, images, input, target, size_average=True):
        # Use the target to reconstruct input image.
        # (batch_size, num_output_units, output_unit_size)
        input = torch.squeeze(input, 3)
        # (batch_size, num_output_units, 1)
        target = torch.unsqueeze(target, 2)
        # (batch_size, output_unit_size, 1)
        masked = torch.matmul(input.transpose(2,1), target)

        # Reconstruct input image.
        masked = masked.view(input.size(0), -1)
        output = self.relu(self.reconstruct0(masked))
        output = self.relu(self.reconstruct1(output))
        output = self.sigmoid(self.reconstruct2(output))
        output = output.view(-1, self.image_channels, self.image_height, self.image_width)

        # Save reconstructed images occasionally.
        if self.reconstructed_image_count % 10 == 0:
            if output.size(1) == 2:
                # handle two-channel images
                zeros = torch.zeros(output.size(0), 1, output.size(2), output.size(3))
                output_image = torch.cat([zeros, output.data.cpu()], dim=1)
            else:
                # assume RGB or grayscale
                output_image = output.data.cpu()
            vutils.save_image(output_image, "reconstruction.png")
        self.reconstructed_image_count += 1

        # The reconstruction loss is the sum squared difference between the input image and reconstructed image.
        # Multiplied by a small number so it doesn't dominate the margin (class) loss.
        error = (output - images).view(output.size(0), -1)
        error = error**2
        error = torch.sum(error, dim=1) * 0.0005

        # Average over batch
        if size_average:
            error = error.mean()

        return error
