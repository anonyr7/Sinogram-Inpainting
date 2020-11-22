import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torchvision.datasets as dset
import torchvision.transforms as transforms

from model import *
from inpaint import *
from utils import *


'''
Please obtain the intermediate results (SIN, SIN_90, SIN_45, SIN_23) from SIN model first.

CUDA_VISIBLE_DEVICES=0 python3 main.py \
--epochs 100 --batchSize 20 -l 0.0001 \
--outdir out \
--input_channel 4 \
--datadir_gt ../Data/reconstructions/train \
--datadir_noisy1 ../Data/SIN_23/train \
--datadir_noisy2 ../Data/SIN_90/train \
--datadir_noisy3 ../Data/SIN_45/train \
--datadir_noisy4 ../Data/SIN/train \
--load -1
'''

parser = argparse.ArgumentParser()
parser.add_argument('--datadir_gt', required=False, default='data', help='directory to ground truth FBP dataset')
parser.add_argument('--datadir_noisy1', required=False, default='data', help='directory to noisy SIN_23 FBP dataset')
parser.add_argument('--datadir_noisy2', required=False, default='data', help='directory to noisy SIN_90 FBP dataset')
parser.add_argument('--datadir_noisy3', required=False, default='data', help='directory to noisy SIN_45 FBP dataset')
parser.add_argument('--datadir_noisy4', required=False, default='data', help='directory to noisy SIN FBP dataset')
parser.add_argument('--outdir', required=False, default='out', help='output dir')
parser.add_argument('--height', required=False, type=int, default=320, help='the height of the input image to network. FIXED.')
parser.add_argument('--width', required=False, type=int, default=320, help='the width of the input image to network. FIXED.')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
parser.add_argument('-b', '--batchSize', metavar='B', type=int, default=64, help='Batch size')
parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, default=0.1, help='Learning rate', dest='lr')
parser.add_argument('--load', dest='load', type=int, default=-1, help='Load model G and D from a .pth file by epoch #. To train from scratch, assign -1. By default checkpoint files from folder outdir/ckpt starting with G_epoch/D_epoch are loaded.')
parser.add_argument('--num_samples', required=False, type=int, default=0, help='default 0: use all dataset available.')
parser.add_argument('--input_channel', required=False, type=int, default=4, help='1 or 4')
args = parser.parse_args()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device {device}')


# LOAD DATASET
if args.datadir_gt is None or args.datadir_noisy1 is None :
    raise ValueError("'datadir_gt' and 'datadir_noisy' parameter is required for dataset")

if args.input_channel == 4:
    dataset = MultiImageFolder(roots=[args.datadir_gt, args.datadir_noisy1, args.datadir_noisy2, args.datadir_noisy3, args.datadir_noisy4],
                            transform=transforms.Compose([
                                transforms.Resize((args.height,args.width)),
                                transforms.Grayscale(), # 1 channel
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                            ]))
elif args.input_channel == 1:
    dataset = MultiImageFolder(roots=[args.datadir_gt, args.datadir_noisy1],
                            transform=transforms.Compose([
                                transforms.Resize((args.height,args.width)),
                                transforms.Grayscale(), # 1 channel
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                            ]))
else:
    print('channel unsupported!')

assert dataset
print(f"Used {args.num_samples if args.num_samples!=0 else len(dataset)} out of {len(dataset)} available data")

if args.num_samples == 0:
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=args.batchSize, num_workers=4)
else:
    weights = []
    for k, v in dataset.class_to_idx.items():
        targets = torch.tensor(dataset.targets)
        target_idx = (targets == v).nonzero()
        weights += [1/len(target_idx)] * len(target_idx)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, args.num_samples)
    dataloader = torch.utils.data.DataLoader(dataset, sampler=sampler, batch_size=args.batchSize, num_workers=4)

    
# LOAD MODELS
netG = UNet(input_nc=args.input_channel, output_nc=1).to(device)
netD = Discriminator(input_nc=1, output_nc=1).to(device)
netG = nn.DataParallel(netG)
netD = nn.DataParallel(netD)
# print(netG)
# print(netD)

# Load Checkpoints
if args.load>=0:
    pathG = os.path.join(args.outdir, 'ckpt/G_epoch'+str(args.load)+'.pth')
    netG.load_state_dict(
        torch.load(pathG, map_location=device)
    )
    print(f'Model G loaded from {pathG}')
    pathD = os.path.join(args.outdir, 'ckpt/D_epoch'+str(args.load)+'.pth')
    netD.load_state_dict(
        torch.load(pathD, map_location=device)
    )
    print(f'Model D loaded from {pathD}')
    
try:
    m = Inpaint(netG=netG, netD=netD, args=args, dataloader=dataloader, device=device)
    m.train()
except KeyboardInterrupt:
    print('Interrupted')
    try:
        sys.exit(0)
    except SystemExit:
        os._exit(0)
