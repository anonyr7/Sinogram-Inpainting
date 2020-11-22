import torch.nn.functional as F
import torch.nn as nn
from torchvision import models

from model_part import *
import functools


class UNet(nn.Module):
    def __init__(self, input_nc=1, output_nc=1, bilinear=True, scale_factor=10):
        super(UNet, self).__init__()
        
        self.scl = UpSample(input_nc, input_nc, bilinear, scale_factor)
        self.inc = DoubleConv(input_nc, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, output_nc)

    def forward(self, x):
        x = self.scl(x)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    
class Discriminator(nn.Module):
    def __init__(self, input_nc=1, output_nc=1):
        super(Discriminator, self).__init__()
        self.noise = GaussianNoise()  # NOT Used!
        self.block1 = nn.Sequential(
            # input is 1 x 320 x 192
            nn.Conv2d(input_nc, 64, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.block2 = nn.Sequential(
            # state size. 64 x 320 x 192
            nn.Conv2d(64, 64*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64*2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.block3 = nn.Sequential(
            # state size. (64*2) x 160 x 96
            nn.Conv2d(64*2, 64*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64*4),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.out = nn.Sequential(
            # state size. (64*4) x 80 x 48
            nn.Conv2d(64*4, output_nc, 3, 1, 1, bias=False)
        )
        
    def forward(self, x):
#         x = self.noise(input)
        x = self.block1(x)
        block1 = x
        x = self.block2(x)
        block2 = x
        x = self.block3(x)
        block3 = x
        x = self.out(x)
        return (block1, block2, block3, x)
#         return (block3, x)


class UpSample(nn.Module):

    def __init__(self, in_channels=1, out_channels=1, bilinear=True, scale_factor=2):
        super().__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=(1, scale_factor), mode='bilinear', align_corners=True)
        else:
            model = [nn.ConvTranspose2d(in_channels, 64, (3,4), (1,2), 1, bias=False),
                     nn.BatchNorm2d(64),
                     nn.ReLU(inplace=True)]
            
            # scale_factor must be exp of 2.
            for i in range(int(math.log2(scale_factor))-1):
                mult = 2 ** i
                model += [nn.ConvTranspose2d(64*mult, 64*mult*2, (3,4), (1,2), 1, bias=False),
                          nn.BatchNorm2d(64*mult*2),
                          nn.ReLU(inplace=True)]
                
            model += [nn.Conv2d(64*mult*2, 64*mult*4, 3, 1, 1, bias=False),
                      nn.BatchNorm2d(64*mult*4),
                      nn.ReLU(inplace=True),
                      nn.Conv2d(64*mult*4, out_channels, 3, 1, 1, bias=False),
                      nn.Tanh()]
            
            self.up = nn.Sequential(*model)

    def forward(self, x):
        return self.up(x)

    
class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        features = models.vgg16(pretrained=True).features

        self.to_relu_1_2 = nn.Sequential() 
        self.to_relu_2_2 = nn.Sequential() 
        self.to_relu_3_3 = nn.Sequential()
        self.to_relu_4_3 = nn.Sequential()

        for x in range(4):
            self.to_relu_1_2.add_module(str(x), features[x])
        for x in range(4, 9):
            self.to_relu_2_2.add_module(str(x), features[x])
        for x in range(9, 16):
            self.to_relu_3_3.add_module(str(x), features[x])
        for x in range(16, 23):
            self.to_relu_4_3.add_module(str(x), features[x])
        
        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = x.repeat(1, 3, 1, 1)  # to rgb
        h = self.to_relu_1_2(x)
        h_relu_1_2 = h
        h = self.to_relu_2_2(h)
        h_relu_2_2 = h
        h = self.to_relu_3_3(h)
        h_relu_3_3 = h
        h = self.to_relu_4_3(h)
        h_relu_4_3 = h
        out = (h_relu_1_2, h_relu_2_2, h_relu_3_3, h_relu_4_3)
        return out
    
    
class GaussianNoise(nn.Module):
    """Gaussian noise regularizer.

    Args:
        sigma (float, optional): relative standard deviation used to generate the
            noise. Relative means that it will be multiplied by the magnitude of
            the value your are adding the noise to. This means that sigma can be
            the same regardless of the scale of the vector.
        is_relative_detach (bool, optional): whether to detach the variable before
            computing the scale of the noise. If `False` then the scale of the noise
            won't be seen as a constant but something to optimize: this will bias the
            network to generate vectors with smaller values.
    """

    def __init__(self, sigma=0.1, is_relative_detach=True):
        super().__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach
        self.register_buffer('noise', torch.tensor(0))

    def forward(self, x):
        if self.sigma != 0:
            scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
            sampled_noise = self.noise.expand(*x.size()).float().normal_() * scale
            x = x + sampled_noise
        return x 
