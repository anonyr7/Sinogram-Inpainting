import torch.nn.functional as F
import torch.nn as nn
from torchvision import models

from model_part import *
import functools


class UNet(nn.Module):
    def __init__(self, input_nc=1, output_nc=1, bilinear=True):
        super(UNet, self).__init__()
        
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
        
    
# class Discriminator(nn.Module):
#     def __init__(self, input_nc=1, output_nc=1):
#         super(Discriminator, self).__init__()

#         self.encode = nn.Sequential(
#             GaussianNoise(),
#             # input is 1 x 320 x 192
#             nn.Conv2d(input_nc, 64, 3, 1, 1, bias=False),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. 64 x 320 x 192
#             nn.Conv2d(64, 64*2, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(64*2),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (64*2) x 160 x 96
#             nn.Conv2d(64*2, 64*4, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(64*4),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (64*4) x 80 x 48
#             nn.Conv2d(64*4, output_nc, 3, 1, 1, bias=False),
#             # new added!
# #             nn.BatchNorm2d(64*8),
# #             # state size. (64*8) x 80 x 48
# #             nn.LeakyReLU(0.2, inplace=True),
# #             nn.Conv2d(64*8, output_nc, 3, 1, 1, bias=False)
#             # output size. 1 x 80 x 48
#         )

#     def forward(self, input):
#         return self.encode(input)

    
class Discriminator(nn.Module):
    def __init__(self, input_nc=1, output_nc=1):
        super(Discriminator, self).__init__()
#         self.noise = GaussianNoise()
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
#         x = self.noise(x)
        x = self.block1(x)
        block1 = x
        x = self.block2(x)
        block2 = x
        x = self.block3(x)
        block3 = x
        x = self.out(x)
        return (block1, block2, block3, x)
    
    
    
# def load_unet(ckpt, device, pretrained=True, **kwargs):
#     model = UNet(**kwargs).to(device)
#     model = nn.DataParallel(model)
    
#     if pretrained:
#         model.load_state_dict(torch.load(ckpt, map_location=device))
#     return model  


# class LossNet(nn.Module):
#     def __init__(self, ckpt, device, **kwargs):
#         super(LossNet, self).__init__()
#         model = load_unet(ckpt, device, pretrained=True, **kwargs)
#         self.inc = model.module.inc
#         self.down1 = model.module.down1
#         self.down2 = model.module.down2
#         self.down3 = model.module.down3
#         self.down4 = model.module.down4
        
#         for param in self.parameters():
#             param.requires_grad = False

#     def forward(self, input):
#         x = self.inc(input)
#         inc = x  # 64 x 320 x 192
#         x = self.down1(x)
#         down1 = x  # 128 x 160 x 96
#         x = self.down2(x)
#         down2 = x  # 256 x 80 x 48
#         x = self.down3(x)
#         down3 = x  # 512 x 40 x 24
#         x = self.down4(x)
#         down4 = x  # 512 x 20 x 12
#         return (inc, down1, down2, down3, down4)
 
    
# class Vgg16(nn.Module):
#     def __init__(self):
#         super(Vgg16, self).__init__()
#         features = models.vgg16(pretrained=True).features

#         self.to_relu_1_2 = nn.Sequential() 
#         self.to_relu_2_2 = nn.Sequential() 
#         self.to_relu_3_3 = nn.Sequential()
#         self.to_relu_4_3 = nn.Sequential()

#         for x in range(4):
#             self.to_relu_1_2.add_module(str(x), features[x])
#         for x in range(4, 9):
#             self.to_relu_2_2.add_module(str(x), features[x])
#         for x in range(9, 16):
#             self.to_relu_3_3.add_module(str(x), features[x])
#         for x in range(16, 23):
#             self.to_relu_4_3.add_module(str(x), features[x])
        
#         # don't need the gradients, just want the features
#         for param in self.parameters():
#             param.requires_grad = False

#     def forward(self, x):
#         x = x.repeat(1, 3, 1, 1)  # to rgb
#         h = self.to_relu_1_2(x)
#         h_relu_1_2 = h
#         h = self.to_relu_2_2(h)
#         h_relu_2_2 = h
#         h = self.to_relu_3_3(h)
#         h_relu_3_3 = h
#         h = self.to_relu_4_3(h)
#         h_relu_4_3 = h
#         out = (h_relu_1_2, h_relu_2_2, h_relu_3_3, h_relu_4_3)
#         return out
    
    
# class UNet(nn.Module):
#     def __init__(self, input_nc=1, output_nc=1, num_downs=4, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, scale_factor=10):

#         super(UNet, self).__init__()
#         self.scl = UpSample(input_nc, output_nc, bilinear=True, scale_factor=scale_factor)
        
#         # construct unet structure
#         unet_block = UnetSkipConnectionBlock(ngf*8, ngf*8, innermost=True)  # add the innermost layer
#         for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
#             unet_block = UnetSkipConnectionBlock(ngf*8, ngf*8, submodule=unet_block, use_dropout=use_dropout)
#         # gradually reduce the number of filters from ngf * 8 to ngf
#         unet_block = UnetSkipConnectionBlock(ngf*4, ngf*8, submodule=unet_block)
#         unet_block = UnetSkipConnectionBlock(ngf*2, ngf*4, submodule=unet_block)
#         unet_block = UnetSkipConnectionBlock(ngf, ngf*2, submodule=unet_block)
#         self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True)  # add the outermost layer

#     def forward(self, input):
#         x = self.scl(input)
#         return self.model(x)


# class UnetSkipConnectionBlock(nn.Module):
#     def __init__(self, outer_nc, inner_nc, input_nc=None,
#                  submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        
#         super(UnetSkipConnectionBlock, self).__init__()
#         self.outermost = outermost
#         if type(norm_layer) == functools.partial:
#             use_bias = norm_layer.func == nn.InstanceNorm2d
#         else:
#             use_bias = norm_layer == nn.InstanceNorm2d
#         if input_nc is None:
#             input_nc = outer_nc
#         downconv = nn.Conv2d(input_nc, inner_nc, 4, 2, 1, bias=use_bias)
#         downrelu = nn.LeakyReLU(0.2, True)
#         downnorm = norm_layer(inner_nc)
#         uprelu = nn.ReLU(True)
#         upnorm = norm_layer(outer_nc)
#         conv1 = nn.Conv2d(input_nc, input_nc, 5, 1, 2, bias=use_bias)
#         conv2 = nn.Conv2d(outer_nc, outer_nc, 5, 1, 2, bias=use_bias)
        
#         if outermost:
#             upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, 4, 2, 1)

#             down = [conv1, downrelu, downconv, downnorm]
#             up = [uprelu, upconv, upnorm, nn.Tanh()]
#             model = down + [submodule] + up
#         elif innermost:
#             upconv = nn.ConvTranspose2d(inner_nc, outer_nc, 5, 1, 2, bias=use_bias)
#             down = [downrelu, downconv]
#             up = [uprelu, upconv, upnorm]
#             model = down + up
#         else:
#             upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, 4, 2, 1, bias=use_bias)
#             downnorm1 = norm_layer(input_nc)
#             down = [downrelu, conv1, downnorm1, downrelu, downconv, downnorm]
#             up = [uprelu, upconv, upnorm, uprelu, conv2, upnorm]

#             if use_dropout:
#                 model = down + [submodule] + up + [nn.Dropout(0.5)]
#             else:
#                 model = down + [submodule] + up

#         self.model = nn.Sequential(*model)

#     def forward(self, x):
#         if self.outermost:
#             return self.model(x)
#         else:   # add skip connections
#             return torch.cat([x, self.model(x)], 1)

# class Discriminator(nn.Module):
#     def __init__(self):
#         super(Discriminator, self).__init__()
        
#         self.main = nn.Sequential(
#             # input is 1 x 320 x 180
#             nn.Conv2d(1, 64, 1, 1, 0, bias=False),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. 64 x 320 x 180
#             nn.Conv2d(64, 64*2, 1, 1, 0, bias=False),
#             nn.BatchNorm2d(64*2),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (64*2) x 320 x 180
#             nn.Conv2d(64*2, 1, 1, 1, 0, bias=False),
#         )

#     def forward(self, input):
#         return self.main(input)
