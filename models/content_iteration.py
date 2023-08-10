import torch
import torch.nn as nn
import torch.nn.functional as F
from models.blocks import *

class ContentIter(nn.Module):
    def __init__(self, content_channels, in_channels):
        super(ContentIter, self).__init__()

        self.content_channels = content_channels
        in_channels = in_channels + 1

        if content_channels == 4:
            self.conv1 = nn.Conv2d(in_channels, 1, kernel_size=1, stride=1, padding=0, bias=False)
            self.conv2 = nn.Conv2d(in_channels, 1, kernel_size=1, stride=1, padding=0, bias=False)
            self.conv3 = nn.Conv2d(in_channels, 1, kernel_size=1, stride=1, padding=0, bias=False)
            self.conv4 = nn.Conv2d(in_channels, 1, kernel_size=1, stride=1, padding=0, bias=False)
        elif content_channels == 5:
            self.conv1 = nn.Conv2d(in_channels, 1, kernel_size=1, stride=1, padding=0, bias=False)
            self.conv2 = nn.Conv2d(in_channels, 1, kernel_size=1, stride=1, padding=0, bias=False)
            self.conv3 = nn.Conv2d(in_channels, 1, kernel_size=1, stride=1, padding=0, bias=False)
            self.conv4 = nn.Conv2d(in_channels, 1, kernel_size=1, stride=1, padding=0, bias=False)
            self.conv5 = nn.Conv2d(in_channels, 1, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, vmf_activations, condition, defined_lv=None):
        out = []

        if defined_lv is not None:
            channel_1_output = defined_lv.detach()
            channel_1 = defined_lv.detach()
        else:
            channel_1_input = torch.cat((vmf_activations, condition), dim=1)
            channel_1_output = self.conv1(channel_1_input)
            channel_1 = channel_1_output
        out.append(channel_1)

        channel_2_input = torch.cat((vmf_activations, channel_1_output.detach()), dim=1)
        channel_2_output = self.conv2(channel_2_input)
        channel_2 = channel_2_output - channel_1_output.detach()
        out.append(channel_2)

        channel_3_input = torch.cat((vmf_activations, channel_2_output.detach()), dim=1)
        channel_3_output = self.conv3(channel_3_input)
        channel_3 = channel_3_output - channel_2_output.detach()
        out.append(channel_3)

        channel_4_input = torch.cat((vmf_activations, channel_3_output.detach()), dim=1)
        channel_4_output = self.conv4(channel_4_input)
        channel_4 = channel_4_output - channel_3_output.detach()
        out.append(channel_4)

        if self.content_channels == 5:
            channel_5_input = torch.cat((vmf_activations, channel_4_output.detach()), dim=1)
            channel_5_output = self.conv5(channel_5_input)
            channel_5 = channel_5_output - channel_4_output.detach()
            out.append(channel_5)

        out = torch.cat(out, dim=1)
        return out