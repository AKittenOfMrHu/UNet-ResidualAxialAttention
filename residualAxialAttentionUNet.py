from residualAxialAttentionBlock import ResidualAxialAttentionBlock, ResidualGatedAxialAttentionBlock

import math
import torch
import torch.nn as nn
import torch.nn.functional as F



class ResidualAxialAttentionUNet(nn.Module):
    def __init__(self, block, in_planes=64, classes=1, head_dim=8, img_size=228, img_planes=3):
        super(ResidualAxialAttentionUNet, self).__init__()
        self.in_planes = in_planes
        self.conv1 = nn.Sequential(
            nn.Conv2d(img_planes, self.in_planes, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )

        self.att1 = self._make_layers(block, 16, 1, stride=1, head_dim=head_dim, pos_dim=img_size//2)
        self.att2 = self._make_layers(block, 32, 2, stride=2, head_dim=head_dim, pos_dim=img_size//2)
        self.att3 = self._make_layers(block, 64, 4, stride=2, head_dim=head_dim, pos_dim=img_size//4)
        self.att4 = self._make_layers(block, 128, 1, stride=2, head_dim=head_dim, pos_dim=img_size//8)
        self.adjust_att = nn.Sequential(
            nn.Conv2d(128*2, 128*2, (3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(128*2)
        )

        self.decoder1 = nn.Sequential(
            nn.Conv2d(128*2, 64*2, (3, 3), padding=(1, 1), bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64*2)
        )
        self.decoder2 = nn.Sequential(
            nn.Conv2d(64*2, 32*2, (3, 3), padding=(1, 1), bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32*2)
        )
        self.decoder3 = nn.Sequential(
            nn.Conv2d(32*2, 16*2, (3, 3), padding=(1, 1), bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16*2)
        )
        self.decoder4 = nn.Sequential(
            nn.Conv2d(16*2, 16*2, (3, 3), padding=(1, 1), bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16*2)
        )
        self.adjust = nn.Conv2d(16*2, classes, (1, 1), bias=False)

    def _make_layers(self, block, planes, layers, stride, head_dim, pos_dim):
        downsample = None
        if self.in_planes != block.expansion or stride > 1:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes*block.expansion, kernel_size=stride, stride=stride, bias=False),
                nn.ReLU(),
                nn.BatchNorm2d(planes*block.expansion)
            )

        blocks = []
        blocks.append(block(self.in_planes, planes, stride, head_dim=head_dim, pos_dim=pos_dim, downsample=downsample))
        self.in_planes = planes*block.expansion

        pos_dim = pos_dim//2 if stride > 1 else pos_dim
        for _ in range(1, layers):
            blocks.append(block(self.in_planes, planes, 1, head_dim, pos_dim))

        return nn.Sequential(*blocks)


    def forward(self, x):
        x1 = self.conv3(self.conv2(self.conv1(x)))
        a1 = self.att1(x1)
        a2 = self.att2(a1)
        a3 = self.att3(a2)
        a4 = self.att4(a3)

        x = F.interpolate(self.adjust_att(a4), scale_factor=2, mode='bilinear')
        x = F.interpolate(self.decoder1(torch.add(x, a4)), scale_factor=2, mode='bilinear')
        x = F.interpolate(self.decoder2(torch.add(x, a3)), scale_factor=2, mode='bilinear')
        x = F.interpolate(self.decoder3(torch.add(x, a2)), scale_factor=2, mode='bilinear')
        x = F.interpolate(self.decoder4(torch.add(x, a1)), scale_factor=2, mode='bilinear')

        out = self.adjust(x)
        return out


def gated_axial_unet(**kwargs):
    model = ResidualAxialAttentionUNet(ResidualGatedAxialAttentionBlock, **kwargs)
    return model


def axial_unet(**kwargs):
    model = ResidualAxialAttentionUNet(ResidualAxialAttentionBlock, **kwargs)
    return model



if __name__ == '__main__':
    img = torch.randn([8, 3, 128, 128])
    print(f'img: {img.shape}')
    model = ResidualAxialAttentionUNet(ResidualAxialAttentionBlock, 64, img_size=128)
    p_y = model(img)
    print(img.shape, p_y.shape)