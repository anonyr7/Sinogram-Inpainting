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
--epochs 100 --batchSize 32 -l 0.0001 \
--outdir out \
--datadir ../../Data/sinograms/train \
--load -1
'''

parser = argparse.ArgumentParser()
parser.add_argument('--datadir', required=False, default='data', help='directory to full-view (target) sinogram dataset')
parser.add_argument('--outdir', required=False, default='out', help='output dir')
parser.add_argument('--height', required=False, type=int, default=320, help='the height of the input image to network. Cannot change for now.')
parser.add_argument('--width', required=False, type=int, default=180, help='the width of the input image to network. Cannot change for now.')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs', dest='epochs')
parser.add_argument('-b', '--batchSize', metavar='B', type=int, nargs='?', default=64, help='Batch size')
parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.1, help='Learning rate', dest='lr')
parser.add_argument('--loadG', dest='loadG', type=str, default=False, help='Load model G from a .pth file')
parser.add_argument('--loadD', dest='loadD', type=str, default=False, help='Load model D from a .pth file')
parser.add_argument('--load', dest='load', type=int, default=0, help='Load model G and D from a .pth file by epoch #')
parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0, help='Percent of the data that is used as validation (0-100)')
parser.add_argument('--ratio', required=False, type=int, default=8)
parser.add_argument('--num_samples', required=False, type=int, default=0, help='default: use all')
args = parser.parse_args()


if args.datadir is None:
    raise ValueError("`datadir` parameter is required for dataset")

dataset = dset.ImageFolder(root=args.datadir,
                            transform=transforms.Compose([
                                transforms.Resize((args.height,args.width)),
                                transforms.Grayscale(), # 1 channel
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device {device}')

netG = UNet(input_nc=1, output_nc=1).to(device)
netD = Discriminator(input_nc=2, output_nc=1).to(device)
netG = nn.DataParallel(netG)
netD = nn.DataParallel(netD)
# print(netG)
# print(netD)

if args.load >= 0:
    path = os.path.join(args.outdir, 'ckpt/G_epoch'+str(args.load)+'.pth')
    netG.load_state_dict(
        torch.load(path, map_location=device)
    )
    print(f'Model G loaded from {path}')
    path = os.path.join(args.outdir, 'ckpt/D_epoch'+str(args.load)+'.pth')
    netD.load_state_dict(
        torch.load(path, map_location=device)
    )
    print(f'Model D loaded from {path}')
    
try:
    m = Inpaint(netG=netG, netD=netD, args=args, dataloader=dataloader, device=device)
    m.train()
except KeyboardInterrupt:
    print('Interrupted')
    try:
        sys.exit(0)
    except SystemExit:
        os._exit(0)
