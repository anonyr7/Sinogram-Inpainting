import torch
import torch.optim as optim
import torchvision.utils as vutils
import torchvision.transforms as transforms

import os
from PIL import Image
import argparse
from model import *
import numpy as np

'''
Run the script:

python3 predict_single.py \
--ckpt out/ckpt/G_epoch110.pth \
--testImage ../../Toy-Dataset/Ground_Truth_sinogram/C/47.png
'''

parser = argparse.ArgumentParser()
parser.add_argument('--height', required=False, default=320, type=int)
parser.add_argument('--width', required=False, default=180, type=int)
parser.add_argument('--testImage', required=False, default='test_sinogram.png')
parser.add_argument('--ckpt', required=True, default='out/ckpt/G_epoch0.pth')
parser.add_argument('--ratio', required=False, type=int, default=8)
args = parser.parse_args()

class Predict():
    def __init__(self, args, image):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.net = UNet(input_nc=1, output_nc=1).to(self.device)
        self.net = nn.DataParallel(self.net)
        
        checkpoint = torch.load(args.ckpt, map_location=self.device)
        self.net.load_state_dict(checkpoint)
        self.net.eval()
        
        self.image = image.to(self.device)
        
        self.mask = self.gen_mask(args.width, args.ratio).to(self.device)
        
        
    def gen_mask(self, w, ratio):
        mask = torch.zeros(w)
        mask[::ratio].fill_(1)
        return mask
    
    
    def gen_x(self, y):
        return self.mask * y
        
        
    def overlay(self, Gx, x):
        result = self.mask * x + (1-self.mask) * Gx
        return result
    
    
    def inpaint(self):
        x = self.gen_x(self.image)
        Gx = self.net(x)
        Gx = self.overlay(Gx, self.image)

        vutils.save_image(Gx, f'result_sinogram.png', normalize=True)


image = Image.open(args.testImage)
transform = transforms.Compose([
                                transforms.Resize((args.height,args.width)),
                                transforms.Grayscale(), # 1 channel
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                              ])
image = transform(image)
image.unsqueeze_(0)
vutils.save_image(image, 'GT_sinogram.png', normalize=True)

p = Predict(args=args, image=image)
p.inpaint()
