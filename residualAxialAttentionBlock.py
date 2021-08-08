from gatedAxialAttention import GatedAxialAttentionBlock
from axialAttention import AxialAttentionBlock

import torch
import torch.nn as nn

def conv1x1(in_channels, out_channels):
    layer = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=(1,1), bias=False),
        nn.ReLU(),
        nn.BatchNorm2d(out_channels)
    )
    return layer



class ResidualAxialAttentionBlock(nn.Module):
    expansion = 2
    def __init__(self, in_planes, planes, stride, head_dim, pos_dim, downsample=None):
        super(ResidualAxialAttentionBlock, self).__init__()
        self.in_conv1x1 = conv1x1(in_planes, planes)
        self.h_att = AxialAttentionBlock(planes, planes, head_dim=head_dim, pos_dim=pos_dim)
        self.w_att = AxialAttentionBlock(planes, planes, head_dim=head_dim, pos_dim=pos_dim)
        self.out_conv1x1 = conv1x1(planes, planes*self.expansion)
        self.pool = nn.AvgPool2d(stride, stride) if stride > 1 else None

        self.downsample = downsample

    def forward(self, x):
        x1 = self.in_conv1x1(x)
        x2 = self.h_att(x1)
        x3 = self.w_att(x2)
        x4 = self.out_conv1x1(x3)
        if self.pool is not None:
            x4 = self.pool(x4)
        if self.downsample is not None:
            x = self.downsample(x)
        out = x + x4
        return out



class ResidualGatedAxialAttentionBlock(nn.Module):
    expansion = 2
    def __init__(self, in_planes, planes, stride, head_dim, pos_dim, downsample=None):
        super(ResidualGatedAxialAttentionBlock, self).__init__()
        self.in_conv1x1 = conv1x1(in_planes, planes)
        self.h_att = AxialAttentionBlock(planes, planes, head_dim=head_dim, pos_dim=pos_dim)
        self.w_att = AxialAttentionBlock(planes, planes, head_dim=head_dim, pos_dim=pos_dim)
        self.out_conv1x1 = conv1x1(planes, planes*self.expansion)
        self.pool = nn.AvgPool2d(stride, stride) if stride > 1 else None

        self.downsample = downsample

    def forward(self, x):
        x1 = self.in_conv1x1(x)
        x2 = self.h_att(x1)
        x3 = self.w_att(x2)
        x4 = self.out_conv1x1(x3)
        if self.pool is not None:
            x4 = self.pool(x4)
        if self.downsample is not None:
            x = self.downsample(x)
        out = x + x4
        return out