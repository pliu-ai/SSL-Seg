"""
2D Conditional U-Net for C3PS.
Condition is concatenated as 2 extra channels at each encoder/decoder level.
Based on the 2D UNet architecture in unet.py.
"""
import torch
import torch.nn as nn


class ConvBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_p):
        super().__init__()
        self.conv_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.conv_conv(x)


class DownBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_p):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock2D(in_channels, out_channels, dropout_p),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpBlock2D(nn.Module):
    def __init__(self, in_channels1, in_channels2, out_channels, dropout_p,
                 bilinear=True):
        super().__init__()
        self.bilinear = bilinear
        if bilinear:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(scale_factor=2, mode='bilinear',
                                  align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels1, in_channels2,
                                         kernel_size=2, stride=2)
        self.conv = ConvBlock2D(in_channels2 * 2, out_channels, dropout_p)

    def forward(self, x1, x2):
        if self.bilinear:
            x1 = self.conv1x1(x1)
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


def _concat_condition(feat, condition):
    """Concatenate 2-channel condition maps to feature tensor (B, C, H, W)."""
    B, _, H, W = feat.shape
    cond = condition.unsqueeze(-1).unsqueeze(-1)
    c = torch.ones(B, 2, H, W, device=feat.device) * cond
    return torch.cat([c, feat], dim=1)


class UNet2DCondition(nn.Module):
    """
    2D U-Net with condition injection at every encoder/decoder level.
    Condition is broadcast as 2 extra channels (+2 to every conv input).
    forward(x, condition) where condition shape is (B, 1).
    """

    def __init__(self, in_chns=1, class_num=2):
        super().__init__()
        ft = [16, 32, 64, 128, 256]
        dp = [0.05, 0.1, 0.2, 0.3, 0.5]

        self.in_conv = ConvBlock2D(in_chns, ft[0], dp[0])
        self.down1 = DownBlock2D(ft[0] + 2, ft[1], dp[1])
        self.down2 = DownBlock2D(ft[1] + 2, ft[2], dp[2])
        self.down3 = DownBlock2D(ft[2] + 2, ft[3], dp[3])
        self.down4 = DownBlock2D(ft[3] + 2, ft[4], dp[4])

        self.up1 = UpBlock2D(ft[4] + 2, ft[3], ft[3], dropout_p=0.0)
        self.up2 = UpBlock2D(ft[3] + 2, ft[2], ft[2], dropout_p=0.0)
        self.up3 = UpBlock2D(ft[2] + 2, ft[1], ft[1], dropout_p=0.0)
        self.up4 = UpBlock2D(ft[1] + 2, ft[0], ft[0], dropout_p=0.0)

        self.out_conv = nn.Conv2d(ft[0] + 2, class_num, kernel_size=3,
                                  padding=1)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x, condition=None):
        if condition is None:
            condition = torch.ones(x.shape[0], 1, device=x.device)

        x0 = self.in_conv(x)
        x0c = _concat_condition(x0, condition)
        x1 = self.down1(x0c)
        x1c = _concat_condition(x1, condition)
        x2 = self.down2(x1c)
        x2c = _concat_condition(x2, condition)
        x3 = self.down3(x2c)
        x3c = _concat_condition(x3, condition)
        x4 = self.down4(x3c)
        x4 = self.dropout(x4)
        x4c = _concat_condition(x4, condition)

        u3 = self.up1(x4c, x3)
        u3c = _concat_condition(u3, condition)
        u2 = self.up2(u3c, x2)
        u2c = _concat_condition(u2, condition)
        u1 = self.up3(u2c, x1)
        u1c = _concat_condition(u1, condition)
        u0 = self.up4(u1c, x0)
        u0 = self.dropout(u0)
        u0c = _concat_condition(u0, condition)

        return self.out_conv(u0c)
