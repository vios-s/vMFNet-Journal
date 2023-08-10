import torch
import torch.nn as nn
import torch.nn.functional as F
from models.blocks import *

class Conv2dBlock(nn.Module):
    def __init__(self, input_dim ,output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero'):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            #self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=True)
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        if norm == 'sn':
            self.conv = SpectralNorm(nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias))
        else:
            self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class ImgCls(nn.Module):
    def __init__(self, norm='bn', activ='relu', pad_type='reflect'):
        super(ImgCls, self).__init__()
        dim = 64
        self.cls_dim = 3
        self.model = []
        self.model += [Conv2dBlock(1, dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
        for i in range(3):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2
        self.model += [Conv2dBlock(dim, dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
        # 288/2-144/2-72/2-36/2-18
        # 512x18x18
        self.model = nn.Sequential(*self.model)
        self.fc1 = nn.Linear(512*9*9, 12*18*18)
        self.norm1 = nn.BatchNorm1d(12*18*18)
        self.activ1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(12*18*18, 2*18*18)
        self.norm2 = nn.BatchNorm1d(2*18*18)
        self.activ2 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(2*18*18, 32)
        self.norm3 = nn.BatchNorm1d(32)
        self.activ3 = nn.ReLU(inplace=True)
        self.fc4 = nn.Linear(32, self.cls_dim)


    def forward(self, x):
        out = self.model(x)
        out = self.fc1(out.view(-1, out.shape[1] * out.shape[2] * out.shape[3]))
        out = self.norm1(out)
        out = self.activ1(out)
        out = self.fc2(out)
        out = self.norm2(out)
        out = self.activ2(out)
        out = self.fc3(out)
        out = self.norm3(out)
        out = self.activ3(out)
        cls = self.fc4(out)
        cls = F.softmax(cls, dim=1)
        return cls

class FeatCls(nn.Module):
    def __init__(self, norm='bn', activ='relu', pad_type='reflect'):
        super(FeatCls, self).__init__()
        dim = 64
        self.cls_dim = 3
        self.model = []
        for i in range(3):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2
        self.model += [Conv2dBlock(dim, dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
        # 288/2-144/2-72/2-36/2-18
        # 512x18x18
        self.model = nn.Sequential(*self.model)
        self.fc1 = nn.Linear(512*9*9, 12*18*18)
        self.norm1 = nn.BatchNorm1d(12*18*18)
        self.activ1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(12*18*18, 2*18*18)
        self.norm2 = nn.BatchNorm1d(2*18*18)
        self.activ2 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(2*18*18, 32)
        self.norm3 = nn.BatchNorm1d(32)
        self.activ3 = nn.ReLU(inplace=True)
        self.fc4 = nn.Linear(32, self.cls_dim)


    def forward(self, x):
        out = self.model(x)
        out = self.fc1(out.view(-1, out.shape[1] * out.shape[2] * out.shape[3]))
        out = self.norm1(out)
        out = self.activ1(out)
        out = self.fc2(out)
        out = self.norm2(out)
        out = self.activ2(out)
        out = self.fc3(out)
        out = self.norm3(out)
        out = self.activ3(out)
        cls = self.fc4(out)
        cls = F.softmax(cls, dim=1)
        return cls

class vMFCls(nn.Module):
    def __init__(self, norm='bn', activ='relu', pad_type='reflect'):
        super(vMFCls, self).__init__()
        dim = 64
        self.cls_dim = 3
        self.model = []
        self.model += [Conv2dBlock(12, dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
        for i in range(3):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2
        # 288/2-144/2-72/2-36/2-18
        # 512x18x18
        self.model = nn.Sequential(*self.model)
        self.fc1 = nn.Linear(512*9*9, 12*18*18)
        self.norm1 = nn.BatchNorm1d(12*18*18)
        self.activ1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(12*18*18, 2*18*18)
        self.norm2 = nn.BatchNorm1d(2*18*18)
        self.activ2 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(2*18*18, 32)
        self.norm3 = nn.BatchNorm1d(32)
        self.activ3 = nn.ReLU(inplace=True)
        self.fc4 = nn.Linear(32, self.cls_dim)


    def forward(self, x):
        out = self.model(x)
        out = self.fc1(out.view(-1, out.shape[1] * out.shape[2] * out.shape[3]))
        out = self.norm1(out)
        out = self.activ1(out)
        out = self.fc2(out)
        out = self.norm2(out)
        out = self.activ2(out)
        out = self.fc3(out)
        out = self.norm3(out)
        out = self.activ3(out)
        cls = self.fc4(out)
        cls = F.softmax(cls, dim=1)
        return cls