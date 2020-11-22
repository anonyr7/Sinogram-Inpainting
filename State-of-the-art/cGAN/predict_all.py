import torch
import torch.optim as optim
import torchvision.utils as vutils
import torchvision.transforms as transforms
import torchvision.datasets as dset

import os
from PIL import Image
import argparse
from model import *
import numpy as np
import math
from torch_radon import Radon
from utils import *


'''
Run this script:

CUDA_VISIBLE_DEVICES=0 python3 predict_all.py \
--datadir ../../Toy-Dataset/Ground_Truth_sinogram \
--outdir ../../Toy-Dataset/CGAN \
--ckpt out/ckpt/G_epoch110.pth \
--class_name N
'''


parser = argparse.ArgumentParser()
parser.add_argument('--datadir', required=True, default='data', help='directory to dataset')
parser.add_argument('--state', required=False, type=str, default='train')
parser.add_argument('--class_name', required=False, type=str, default='C')
parser.add_argument('--height', required=False, default=320, type=int)
parser.add_argument('--width', required=False, default=180, type=int)
parser.add_argument('-b', '--bs', type=int, default=10, help='Batch size')
parser.add_argument('--ckpt', dest='ckpt', required=True, help='Load model G and D from a .pth file')
parser.add_argument('--outdir', required=True, default='out', help='output dir')
parser.add_argument('--angles', required=False, type=int, default=23)
parser.add_argument('--num_samples', required=False, type=int, default=0, help='default: use all')
args = parser.parse_args()


class Predict():
    def __init__(self, args, dataloader):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.args = args
        self.dataloader = dataloader
            
        self.net = UNet(input_nc=1, output_nc=1).to(self.device)
        self.net = nn.DataParallel(self.net)
        
        pathG = os.path.join(args.ckpt)
        self.net.load_state_dict(torch.load(pathG, map_location=self.device))
        self.net.eval()
        
        self.gen_mask()
        
        angles = np.linspace(0, np.pi, 180, endpoint=False)
        self.radon = Radon(args.height, angles, clip_to_circle=True)
        
        
    def gen_mask(self):
        self.mask = torch.zeros(180).to(self.device)
        self.mask[::8].fill_(1)  # 180
            
            
    def gen_x(self, y):
        return self.mask * y
    
    
    def crop_sinogram(self, x):
        return x[:,:,:,6:-6]
        
        
    def overlay(self, Gx, x):
        result = self.mask * x + (1-self.mask) * Gx
        return result
    
    
    def inpaint(self):
        for i, data in enumerate(self.dataloader):
            y = data[0].to(self.device)  # 320 x 180

            x = self.gen_x(y)  # input, 320 x 23
            Gx = self.net(x)
            
            Gx = self.overlay(Gx, y)
            
            # FBP
            Gx = normalize(Gx)  # 0~1
            fbp_Gx = self.radon.backprojection(self.radon.filter_sinogram(Gx.permute(0,1,3,2)))
            
            print(f'Saving images for batch {i}')

            for j in range(y.size()[0]):
#                 vutils.save_image(Gx[j,0], f'{self.args.outdir}/{class_name}/{fnames[i*self.args.bs+j]}', normalize=True)  # to 0~255
                vutils.save_image(fbp_Gx[j,0], f'{self.args.outdir}/{class_name}/{fnames[i*self.args.bs+j]}', normalize=True)

                

if args.datadir is None:
    raise ValueError("`datadir` parameter is required for dataset")

datadir = args.datadir
dataset = dset.ImageFolder(root=datadir,
                            transform=transforms.Compose([
                                transforms.Resize((args.height,args.width)),
                                transforms.Grayscale(), # 1 channel
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),  # has already been 0~1 before this, to -1~1
                            ]))

assert dataset
print(f"Used {args.num_samples if args.num_samples!=0 else len(dataset)} out of {len(dataset)} available data")

class_name = args.class_name
state = args.state
class_idx = dataset.class_to_idx[class_name]
print(class_name, class_idx)
targets = torch.tensor(dataset.targets)
target_idx = np.nonzero(targets == class_idx)
print(len(target_idx))
subset = torch.utils.data.Subset(dataset, target_idx)
sampler = torch.utils.data.sampler.SequentialSampler(subset)
dataloader = torch.utils.data.DataLoader(subset, sampler=sampler, batch_size=args.bs)

# make sure file sequence is the same
fnames = sorted(sorted(os.walk(os.path.join(args.datadir, class_name), followlinks=True))[0][2])

try:
    if not os.path.exists(os.path.join(args.outdir, class_name)):
        os.makedirs(os.path.join(args.outdir, class_name))
        print('Created CGAN directory')
    else:
        print('saving to', os.path.join(args.outdir, class_name))
except OSError:
    pass

p = Predict(args=args, dataloader=dataloader)
p.inpaint()
