import torch.nn.functional as F
import torch.nn as nn

from model_part import *
import functools

class UNet(nn.Module):
    def __init__(self, input_nc=1, output_nc=1):
        super(UNet, self).__init__()
        
        # CL64, 1 x 320 x 180
        self.inc = Down(input_nc, 64, batch_norm=False)
        # CBL128, 64 x 160 x 90
        self.down1 = Down(64, 128)
        # CBL256, 128 x 80 x 45
        self.down2 = Down(128, 256)
        # CBL512, 256 x 40 x 22
        self.down3 = Down(256, 512)
        # DBR256, 512 x 20 x 11
        self.up1 = Up(512, 256)
        # DBR128, 512 x 40 x 22
        self.up2 = Up(512, 128)
        # DBR64, 256 x 80 x 45
        self.up3 = Up(256, 64)
        # D1, 128 x 160 x 90
        self.outc = OutConv(128, output_nc)
        # 1 x 320 x 180

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)
        return logits


class Discriminator(nn.Module):
    def __init__(self, input_nc=1, output_nc=1):
        super(Discriminator, self).__init__()
        
        self.main = nn.Sequential(
            # input is 1 x 320 x 180
            nn.Conv2d(input_nc, 64, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. 64 x 320 x 180
            nn.Conv2d(64, 64*2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64*2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (64*2) x 320 x 180
            nn.Conv2d(64*2, output_nc, 3, 1, 1, bias=False),
        )

    def forward(self, input):
        return self.main(input)