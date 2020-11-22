import os
import sys
import argparse
from PIL import Image

import torch
import torchvision.utils as vutils
import torchvision.transforms as transforms

from gradient_descent_network import *
from neumann_network import *

'''
CUDA_VISIBLE_DEVICES=0 python3 predict_single.py \
--testImage ../../Toy-Dataset/Ground_Truth_sinogram/C/47.png \
--outdir out \
--load 52 \
--net NN
'''

parser = argparse.ArgumentParser()
parser.add_argument('--testImage', required=True, help='test full-view sinogram image')
parser.add_argument('--outdir', required=False, default='out', help='ckpt dir')
parser.add_argument('--net', required=False, type=str, default='NN', help='GD: Unrolled Gradiant Descent; NN: Neumann Network')
parser.add_argument('--load', dest='load', type=int, required=True, default=-1, help='Load model from a .pth file by epoch #')
parser.add_argument('--blocks', type=int, default=6, dest='blocks', help='Number of blocks (iterations)')
parser.add_argument('--height', required=False, default=320, type=int)
parser.add_argument('--width', required=False, default=180, type=int)
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device {device}.')

transform = transforms.Compose([
                                transforms.Resize((args.height,args.width)),
                                transforms.Grayscale(), # 1 channel
                                transforms.ToTensor(),
                              ])

image = Image.open(args.testImage)
assert image
image = transform(image)
image.unsqueeze_(0)
vutils.save_image(image, 'GT_sinogram.png', normalize=True)

try:
    if args.net == 'GD':
        m = GradientDescentNet(args=args, dataloader=None, device=device)
    elif args.net == 'NN':
        m = NeumannNet(args=args, dataloader=None, device=device)
    results = m.test(image)
    vutils.save_image(results, f'test_result.png', normalize=True)
except KeyboardInterrupt:
    print('Interrupted')
    try:
        sys.exit(0)
    except SystemExit:
        os._exit(0)