import torch.nn as nn
from models.blocks import *

class Discriminator(nn.Module):
    def __init__(self, num_channels):
        super(Discriminator, self).__init__()

        self.num_channels = num_channels
        dim = 16

        self.model = []
        self.model += [conv_bn_lrelu(self.num_channels, dim, 4, 2, 1)] # 16x144x144
        for i in range(4):
            self.model += [conv_bn_lrelu(dim, 2 * dim, 4, 2, 1)]
            dim *= 2
        for i in range(3):
            self.model += [conv_bn_lrelu(dim, dim, 4, 2, 1)]
        self.model = nn.Sequential(*self.model)
        # 256
        self.fc1 = nn.Linear(256, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        batch_size = x.size(0)
        out = self.model(x)
        out = out.view(batch_size, -1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out