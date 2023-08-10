import torch.nn as nn
from models.blocks import *

class Heartor(nn.Module):
    def __init__(self, num_channels=1):
        super(Heartor, self).__init__()

        self.num_channels = 3
        dim = 32

        self.model = []
        self.model += [conv_bn_lrelu(self.num_channels, dim, 4, 2, 1)] # 16x77x77
        for i in range(5):
            self.model += [conv_bn_lrelu(dim, 2 * dim, 4, 2, 1)]
            dim *= 2
        #256x4x4
        self.model = nn.Sequential(*self.model)
        # 256
        self.fc1 = nn.Linear(1024*4*4, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        batch_size = x.size(0)
        # x = x[:, 0, :, :].unsqueeze(1)
        out = self.model(x)
        out = out.view(batch_size, -1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out