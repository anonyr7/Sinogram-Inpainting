import torch
import torch.optim as optim
import torchvision.utils as vutils
import torchvision.transforms as transforms

import os
from PIL import Image
import argparse
import numpy as np
from torch_radon import Radon
import math

from model import *
from utils import *


'''
CUDA_VISIBLE_DEVICES=0 python3 predict_single.py \
--testImage ../Toy-Dataset/Ground_Truth_sinogram/C/47.png \
--ckpt out/ckpt/G_epoch49.pth
'''

parser = argparse.ArgumentParser()
parser.add_argument('--testImage', required=False, help='Path to a test .png full-view sinogram image.')
parser.add_argument('--ckpt', required=True, type=str, help='Path of the generator checkpoint file.')
parser.add_argument('--height', required=False, default=320, type=int, help='FIXED')
parser.add_argument('--width', required=False, default=180, type=int, help='FIXED')
parser.add_argument('--angles', required=False, type=int, default=23, help='Known angle number. FIXED!')
parser.add_argument('--twoends', required=False, action='store_false', default=True, help='Whether use two-ends preprocessing. Adjust according to trained ckpt file. Default True.')
args = parser.parse_args()

class Predict():
    def __init__(self, args, image):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if args.twoends:
            factor = 192 / (args.angles+2)  # 7.68
        else:
            factor = 180 / args.angles  # 7.826086956521739
    
        self.net = UNet(input_nc=1, output_nc=1, scale_factor=factor).to(self.device)
        self.net = nn.DataParallel(self.net)
        pathG = os.path.join(args.ckpt)
        self.net.load_state_dict(torch.load(pathG, map_location=self.device))
        self.net.eval()
        
        self.image = image.to(self.device)
        self.twoends = args.twoends
        self.mask = self.gen_mask().to(self.device)
        
        # Radon Operator
        angles = np.linspace(0, np.pi, 180, endpoint=False)
        self.radon = Radon(args.height, angles, clip_to_circle=True)
        
        
    def gen_mask(self):
        mask = torch.zeros(180)
        mask[::8].fill_(1)  # 180
        if self.twoends:
            mask = torch.cat((mask[-6:], mask, mask[:6]), 0) # 192
        return mask
    
    
    def append_twoends(self, y):
        front = torch.flip(y[:,:,:,:6], [2])
        back = torch.flip(y[:,:,:,-6:], [2])
        return torch.cat((back, y, front), 3)
    
    
    def gen_sparse(self, y):
        return y[:,:,:,self.mask==1]
    
    
    def crop_sinogram(self, x):
        return x[:,:,:,6:-6]
        
        
    def inpaint(self):
        y = self.image  # 320 x 180
        
        # Two-Ends Preprocessing
        if self.twoends:
            y = self.append_twoends(y)  # 320 x 192
        
        # Generate Sparse-view Image, forward model
        x = self.gen_sparse(y)
        Gx = self.net(x)
        
        # Crop Two-Ends
        if self.twoends:
            Gx = self.crop_sinogram(Gx)
        
        # FBP Reconstruction
        Gx = normalize(Gx)  # 0~1
        fbp_Gx = self.radon.backprojection(self.radon.filter_sinogram(Gx.permute(0,1,3,2)))
        
        # Save Results
        vutils.save_image(fbp_Gx, 'result_reconstruction.png', normalize=True)
        vutils.save_image(Gx, 'result_sinogram.png', normalize=True)


image = Image.open(args.testImage)
transform = transforms.Compose([
                                transforms.Resize((args.height, args.width)),
                                transforms.Grayscale(), # 1 channel
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                              ])
image = transform(image)
image.unsqueeze_(0)
vutils.save_image(image, 'GT_sinogram.png', normalize=True)

p = Predict(args=args, image=image)
p.inpaint()