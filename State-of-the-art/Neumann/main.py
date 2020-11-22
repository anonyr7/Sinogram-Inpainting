import argparse
import os
import sys

import torch
import torchvision.transforms as transforms
import torchvision.datasets as dset

from gradient_descent_network import *
from neumann_network import *


parser = argparse.ArgumentParser()
parser.add_argument('--datadir', required=True, default='data', help='directory to sinogram dataset')
parser.add_argument('--outdir', required=False, default='out', help='output dir')
parser.add_argument('--epochs', type=int, default=100, dest='epochs', help='Number of epochs to train')
parser.add_argument('--blocks', type=int, default=6, dest='blocks', help='Number of blocks (iterations)')
parser.add_argument('--bs', type=int, default=64, help='Batch size')
parser.add_argument('-lr', type=float, default=1e-5, dest='lr', help='Learning rate')
parser.add_argument('--height', required=False, type=int, default=320, help='the height of the input image to network.')
parser.add_argument('--width', required=False, type=int, default=180, help='the width of the input image to network.')
parser.add_argument('--net', required=False, type=str, default='NN', help='GD: Unrolled Gradiant Descent; NN: Neumann Network')
parser.add_argument('--load', dest='load', type=int, default=-1, help='Load model from a .pth file by epoch #')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device {device}.')

dataset = dset.ImageFolder(root=args.datadir,
                            transform=transforms.Compose([  # to 0~1
                                transforms.Resize((args.height,args.width)),
                                transforms.Grayscale(),
                                transforms.ToTensor(),
#                                 transforms.Normalize((0.5,), (0.5,))
                            ]))
assert dataset
print(f"Dataset contains {len(dataset)} images.")
dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=args.bs, num_workers=4)


try:
    if not os.path.exists(os.path.join(args.outdir, 'ckpt')):
        os.makedirs(os.path.join(args.outdir, 'ckpt'))
        print('Created checkpoint directory')
except OSError:
    pass


try:
    if args.net == 'GD':
        m = GradientDescentNet(args=args, dataloader=dataloader, device=device)
    elif args.net == 'NN':
        m = NeumannNet(args=args, dataloader=dataloader, device=device)
    m.train()
except KeyboardInterrupt:
    print('Interrupted')
    try:
        sys.exit(0)
    except SystemExit:
        os._exit(0)
