import torch
import torch.optim as optim
import torchvision.utils as vutils
import torchvision.transforms as transforms

import os
from PIL import Image
import argparse
from model import *
import numpy as np
from utils import *

'''
CUDA_VISIBLE_DEVICES=0 python3 predict_single.py \
--testImage_noisy1 ../Toy-Dataset/SIN_23/C/47.png \
--testImage_noisy2 ../Toy-Dataset/SIN_90/C/47.png \
--testImage_noisy3 ../Toy-Dataset/SIN_45/C/47.png \
--testImage_noisy4 ../Toy-Dataset/SIN/C/47.png \
--testImage_gt ../Toy-Dataset/Ground_Truth_reconstruction/C/47.png \
--ckpt out/ckpt/G_epoch70.pth
'''

parser = argparse.ArgumentParser()
parser.add_argument('--testImage_noisy1', required=True, help='SIN_23 test reconstruction image')
parser.add_argument('--testImage_noisy2', required=True, help='SIN_90 test reconstruction image')
parser.add_argument('--testImage_noisy3', required=True, help='SIN_45 test reconstruction image')
parser.add_argument('--testImage_noisy4', required=True, help='SIN test reconstruction image')
parser.add_argument('--testImage_gt', required=True, help='test ground truth reconstruction image')
parser.add_argument('--ckpt', required=True, default='out/ckpt/G_epoch0.pth')
parser.add_argument('--height', required=False, default=320, type=int, help='FIXED')
parser.add_argument('--width', required=False, default=320, type=int, help='FIXED')
args = parser.parse_args()

class Predict():
    def __init__(self, args, image):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.net = UNet(input_nc=4, output_nc=1).to(self.device)
        self.net = nn.DataParallel(self.net)
        
        checkpoint = torch.load(args.ckpt, map_location=self.device)
        self.net.load_state_dict(checkpoint)
        self.net.eval()
        
        self.image = image
        

    def inpaint(self):
        x1 = self.image[0].to(self.device)
        x2 = self.image[1].to(self.device)
        x3 = self.image[2].to(self.device)
        x4 = self.image[3].to(self.device)
        x = torch.cat((x1,x2,x3,x4),1)
        Gx = self.net(x)
        
        vutils.save_image(Gx, 'result_reconstruction.png', normalize=True)

image_gt = Image.open(args.testImage_gt)
image1 = Image.open(args.testImage_noisy1)
image2 = Image.open(args.testImage_noisy2)
image3 = Image.open(args.testImage_noisy3)
image4 = Image.open(args.testImage_noisy4)

transform = transforms.Compose([
                                transforms.Resize((args.height,args.width)),
                                transforms.Grayscale(), # 1 channel
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                              ])

image_gt = transform(image_gt)
image_gt.unsqueeze_(0)
vutils.save_image(image_gt, 'GT_reconstruction.png', normalize=True)

image1 = transform(image1)
image1.unsqueeze_(0)
image2 = transform(image2)
image2.unsqueeze_(0)
image3 = transform(image3)
image3.unsqueeze_(0)
image4 = transform(image4)
image4.unsqueeze_(0)

p = Predict(args=args, image=[image1, image2, image3, image4])
p.inpaint()
