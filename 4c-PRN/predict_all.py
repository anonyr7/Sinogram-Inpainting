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
from utils import *


'''
CUDA_VISIBLE_DEVICES=0 python3 predict_all.py \
--datadir_noisy1 ../Toy-Dataset/SIN_23 \
--datadir_noisy2 ../Toy-Dataset/SIN_90 \
--datadir_noisy3 ../Toy-Dataset/SIN_45 \
--datadir_noisy4 ../Toy-Dataset/SIN \
--outdir ../Toy-Dataset/SIN-4c-PRN \
--ckpt out/ckpt/G_epoch70.pth \
--class_name C
'''

parser = argparse.ArgumentParser()
parser.add_argument('--datadir_noisy1', required=True, default='data/SIN_23', help='directory to dataset SIN_23')
parser.add_argument('--datadir_noisy2', required=True, default='data/SIN_90', help='directory to dataset SIN_90')
parser.add_argument('--datadir_noisy3', required=True, default='data/SIN_45', help='directory to dataset SIN_45')
parser.add_argument('--datadir_noisy4', required=True, default='data/SIN', help='directory to dataset SIN')
parser.add_argument('--class_name', required=True, type=str, default='C', help='Data Category: C/L/N')
parser.add_argument('--ckpt', required=True, help='Load model G from a .pth file')
parser.add_argument('--outdir', required=True, default='out', help='output directory')
parser.add_argument('-b', '--bs', type=int, default=5, help='Batch size')
parser.add_argument('--height', required=False, default=320, type=int, help='FIXED')
parser.add_argument('--width', required=False, default=320, type=int, help='FIXED')
parser.add_argument('--num_samples', required=False, type=int, default=0, help='default: use all')
args = parser.parse_args()


class Predict():
    def __init__(self, args, dataloader):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.args = args
        self.dataloader = dataloader
            
        self.net = UNet(input_nc=4).to(self.device)
        self.net = nn.DataParallel(self.net)
        self.net.load_state_dict(torch.load(args.ckpt, map_location=self.device))
        self.net.eval()


    def inpaint(self):
        for i, data in enumerate(self.dataloader):
            # Load Inputs
            x1 = data[0].to(self.device)  # 320 x 320
            x2 = data[1].to(self.device)
            x3 = data[2].to(self.device)
            x4 = data[3].to(self.device)
            x = torch.cat((x1,x2,x3,x4),1)
            
            # Forward
            Gx = self.net(x)
            
            print(f'Saving images for batch {i}')
            
            for j in range(Gx.size()[0]):
                vutils.save_image(Gx[j,0], f'{self.args.outdir}/{class_name}/{fnames[i*self.args.bs+j]}', normalize=True)  # to 0~255
            

# Load Dataset
dataset = MultiImageFolder(roots=[args.datadir_noisy1, args.datadir_noisy2, args.datadir_noisy3, args.datadir_noisy4],
                            transform=transforms.Compose([
                                transforms.Resize((args.height,args.width)),
                                transforms.Grayscale(), # 1 channel
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
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
fnames = sorted(sorted(os.walk(os.path.join(args.datadir_noisy1, class_name), followlinks=True))[0][2])


# Create Result Folders
try:
    path = os.path.join(args.outdir, class_name)
    if not os.path.exists(path):
        os.makedirs(path)
        print(f'Created {path} directory.')
    else:
        print(f'Saving to {path} directory.')
except OSError:
    pass


# Predict All
p = Predict(args=args, dataloader=dataloader)
p.inpaint()