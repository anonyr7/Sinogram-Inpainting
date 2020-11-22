import torch
import torch.optim as optim
import torchvision.utils as vutils
import torchvision.transforms as transforms
import torchvision.datasets as dset

import os
from PIL import Image
import argparse
import numpy as np
import math
from torch_radon import Radon
from model import *
from utils import *


'''
CUDA_VISIBLE_DEVICES=0 python3 predict_all.py \
--datadir ../Toy-Dataset/Ground_Truth_sinogram \
--outdir ../Toy-Dataset/SIN \
--ckpt out/ckpt/G_epoch49.pth \
--class_name N
'''


parser = argparse.ArgumentParser()
parser.add_argument('--datadir', required=True, default='data', help='directory to full-view sinogram dataset')
parser.add_argument('--class_name', required=True, type=str, default='C', help='subfolder name. E.g.: C, L or N.')
parser.add_argument('--ckpt', required=True, help='Load model G from a .pth file')
parser.add_argument('--outdir', required=True, default='out', help='output directory')
parser.add_argument('--height', required=False, default=320, type=int, help='FIXED')
parser.add_argument('--width', required=False, default=180, type=int, help='FIXED')
parser.add_argument('-b', '--bs', type=int, default=10, help='Batch size')
parser.add_argument('--angles', required=False, type=int, default=23, help='Known angle number. FIXED!')
parser.add_argument('--twoends', required=False, action='store_false', default=True, help='Whether use two-ends preprocessing. Adjust according to trained ckpt file. Default True.')
parser.add_argument('--num_samples', required=False, type=int, default=0, help='default: use all')
args = parser.parse_args()


class Predict():
    def __init__(self, args, dataloader):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.args = args
        self.dataloader = dataloader
        
        if args.twoends:
            factor = 192 / (args.angles+2)  # 7.68
        else:
            factor = 180 / args.angles  # 7.826086956521739
            
        self.net = UNet(input_nc=1, output_nc=1, scale_factor=factor).to(self.device)
        self.net = nn.DataParallel(self.net)
        pathG = os.path.join(args.ckpt)
        self.net.load_state_dict(torch.load(pathG, map_location=self.device))
        self.net.eval()
        
        self.gen_mask()
        
        # Radon Operator for different downsampling factors
        angles = np.linspace(0, np.pi, 180, endpoint=False)
        self.radon = Radon(args.height, angles, clip_to_circle=True)
        self.radon23 = Radon(args.height, angles[::8], clip_to_circle=True)
        self.radon45 = Radon(args.height, angles[::4], clip_to_circle=True)
        self.radon90 = Radon(args.height, angles[::2], clip_to_circle=True)
        
        
    def gen_mask(self):
        mask = torch.zeros(180)
        mask[::8].fill_(1)  # 180
        if self.args.twoends:
            self.mask = torch.cat((mask[-6:], mask, mask[:6]), 0).to(self.device) # 192
        self.mask_sparse = mask
    
    
    def append_twoends(self, y):
        front = torch.flip(y[:,:,:,:6], [2])
        back = torch.flip(y[:,:,:,-6:], [2])
        return torch.cat((back, y, front), 3)
    
    
    def gen_input(self, y, mask):
        return y[:,:,:,mask==1]
    
    
    def crop_sinogram(self, x):
        return x[:,:,:,6:-6]

    
    def inpaint(self):
        for i, data in enumerate(self.dataloader):
            y = data[0].to(self.device)  # 320 x 180
            
            # Two-Ends Preprocessing
            if self.args.twoends:
                y_TE = self.append_twoends(y)  # 320 x 192

            # Forward Model
            x = self.gen_input(y_TE, self.mask)  # input, 320 x 25
            Gx = self.net(x)  # 320 x 192
            
            # Crop Two-Ends
            if self.args.twoends:
                Gx = self.crop_sinogram(Gx)  # 320 x 180
            
            # FBP Reconstruction
            Gx = normalize(Gx)  # 0~1
            fbp_Gx = self.radon.backprojection(self.radon.filter_sinogram(Gx.permute(0,1,3,2)))  # 320 x 320
            
            # FBP for downsampled sinograms
            Gx1 = Gx[:,:,:,::2] # 320 x 90
            Gx1 = normalize(Gx1)  # 0~1
            fbp_Gx1 = self.radon90.backprojection(self.radon90.filter_sinogram(Gx1.permute(0,1,3,2)))
            
            Gx2 = Gx[:,:,:,::4] # 320 x 45
            Gx2 = normalize(Gx2)  # 0~1
            fbp_Gx2 = self.radon45.backprojection(self.radon45.filter_sinogram(Gx2.permute(0,1,3,2)))
            
            sparse = y[:,:,:,::8]  # 320 x 23, original sparse-view sinogram
            sparse = normalize(sparse)  # 0~1
            fbp_sparse = self.radon23.backprojection(self.radon23.filter_sinogram(sparse.permute(0,1,3,2)))
            
            print(f'Saving images for batch {i}')

            for j in range(y.size()[0]):
#                 vutils.save_image(Gx[j,0], f'{self.args.outdir}/{class_name}/{fnames[i*self.args.bs+j]}', normalize=True)
                vutils.save_image(fbp_Gx[j,0], f'{self.args.outdir}/{class_name}/{fnames[i*self.args.bs+j]}', normalize=True)
                vutils.save_image(fbp_Gx1[j,0], f'{self.args.outdir}_90/{class_name}/{fnames[i*self.args.bs+j]}', normalize=True)
                vutils.save_image(fbp_Gx2[j,0], f'{self.args.outdir}_45/{class_name}/{fnames[i*self.args.bs+j]}', normalize=True)
                vutils.save_image(fbp_sparse[j,0], f'{self.args.outdir}_23/{class_name}/{fnames[i*self.args.bs+j]}', normalize=True)
                

# DATALOADER
if args.datadir is None:
    raise ValueError("`datadir` parameter is required for dataset")


dataset = dset.ImageFolder(root=args.datadir,
                            transform=transforms.Compose([
                                transforms.Resize((args.height,args.width)),
                                transforms.Grayscale(), # 1 channel
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),  # has already been 0~1 before this, to -1~1
                            ]))

assert dataset
print(f"Used {args.num_samples if args.num_samples!=0 else len(dataset)} out of {len(dataset)} available data")

class_name = args.class_name
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


# Create Directory for results
try:
    path = os.path.join(args.outdir, class_name)
    if not os.path.exists(path):
        os.makedirs(path)
        print(f'Created {path} directory')
    else:
        print(f'saving to {path}')
    
    # Directory for downsampled images
    path = os.path.join(args.outdir+'_23', class_name)
    if not os.path.exists(path):
        os.makedirs(path)
        print(f'Created {path} directory')
    else:
        print(f'saving to {path}')
        
    # Directory for downsampled images
    path = os.path.join(args.outdir+'_45', class_name)
    if not os.path.exists(path):
        os.makedirs(path)
        print(f'Created {path} directory')
    else:
        print(f'saving to {path}')
        
    # Directory for downsampled images
    path = os.path.join(args.outdir+'_90', class_name)
    if not os.path.exists(path):
        os.makedirs(path)
        print(f'Created {path} directory')
    else:
        print(f'saving to {path}')
except OSError:
    pass

# Predict All
p = Predict(args=args, dataloader=dataloader)
p.inpaint()
