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


'''
CUDA_VISIBLE_DEVICES=0 python3 main.py \
--epochs 100 --batchSize 20 -l 0.0001 \
--datadir ../Data/sinograms/train \
--outdir out \
--load -1
'''

parser = argparse.ArgumentParser()
parser.add_argument('--datadir', required=True, default='data', help='directory to full-view (target) sinogram dataset. Images are in subfolders C, N, L or others.')
parser.add_argument('--outdir', required=True, default='out', help='output dir')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
parser.add_argument('-b', '--batchSize', type=int, default=64, help='Batch size')
parser.add_argument('-l', '--learning-rate', type=float, default=0.0001, dest='lr', help='Learning rate')
parser.add_argument('--load', dest='load', type=int, default=-1, help='Load model G and D from a .pth file by epoch #. To train from scratch, assign -1. By default checkpoint files from folder outdir/ckpt starting with G_epoch/DG_epoch/DL_epoch are loaded.')
parser.add_argument('--height', required=False, type=int, default=320, help='the height of the input image to network. FIXED.')
parser.add_argument('--width', required=False, type=int, default=180, help='the width of the input image to network. FIXED.')
parser.add_argument('--angles', required=False, type=int, default=23, help='#angles for sparse-view sinogram. FIXED.')
parser.add_argument('--num_samples', required=False, type=int, default=0, help='default: use all dataset available.')
parser.add_argument('--twoends', required=False, action='store_false', default=True, help='Whether use two-ends preprocessing. Adjust according to trained ckpt file. Default True.')
parser.add_argument('--mode', required=False, type=str, default='DP', help='DP/vgg/NP. DP: Discriminator Perceptual Loss; vgg: VGG16 perceptual loss; NP or others: No perceptual loss.')
parser.add_argument('--log_fn', required=False, type=str, default='log')
args = parser.parse_args()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device {device}')
print('Perceptual mode: ', args.mode)

# Load Dataset
if args.datadir is None:
    raise ValueError("`datadir` parameter is required for dataset")

dataset = dset.ImageFolder(root=args.datadir,
                            transform=transforms.Compose([
                                transforms.Resize((args.height,args.width)),
                                transforms.Grayscale(), # 1 channel
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),  # has already been 0~1 before this
                            ]))
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


# Load Models
if args.twoends:
    factor = 192 / (args.angles+2)  # 7.68
else:
    factor = 180 / args.angles  # 7.826086956521739

netG = UNet(input_nc=1, output_nc=1, scale_factor=factor).to(device)
netG = nn.DataParallel(netG)
print(netG)
netDG = Discriminator().to(device)
netDG = nn.DataParallel(netDG)
print(netDG)
netDL = Discriminator().to(device)
netDL = nn.DataParallel(netDL)
print(netDL)
net=[netG,netDG,netDL]
if args.mode == 'vgg':
    print('use vgg')
    netLoss = Vgg16().to(device)
    netLoss = nn.DataParallel(netLoss)
    net.append(netLoss)

    
# Load Checkpoints
if args.load>=0: 
    pathG = os.path.join(args.outdir, 'ckpt/G_epoch'+str(args.load)+'.pth')
    netG.load_state_dict(
        torch.load(pathG, map_location=device)
    )
    print(f'Model G loaded from {pathG}')

    pathDG = os.path.join(args.outdir, 'ckpt/DG_epoch'+str(args.load)+'.pth')
    netDG.load_state_dict(
        torch.load(pathDG, map_location=device)
    )
    print(f'Model DG loaded from {pathDG}')
    
    pathDL = os.path.join(args.outdir, 'ckpt/DL_epoch'+str(args.load)+'.pth')
    netDL.load_state_dict(
        torch.load(pathDL, map_location=device)
    )
    print(f'Model DL loaded from {pathDL}')
    
    
try:
    m = Inpaint(net=net, args=args, dataloader=dataloader, device=device)
    m.train()
except KeyboardInterrupt:
    print('Interrupted')
    try:
        sys.exit(0)
    except SystemExit:
        os._exit(0)
