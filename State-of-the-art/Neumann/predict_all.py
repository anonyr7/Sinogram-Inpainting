# from torch_radon import Radon
import torch
import numpy as np
import torchvision.utils as vutils
import torchvision.datasets as dset
import torchvision.transforms as transforms
import os
import argparse

from utils import *
from neumann_network import *

'''
CUDA_VISIBLE_DEVICES=0 python3 predict_all.py \
--datadir ../../Toy-Dataset/Ground_Truth_sinogram \
--outdir out \
--saveto ../../Toy-Dataset/Neumann \
--load 52 \
--net NN \
--class_name N
'''

parser = argparse.ArgumentParser()
parser.add_argument('--datadir', required=True, default='data', help='directory to sinogram dataset')
parser.add_argument('--outdir', required=False, default='out', help='ckpt dir')
parser.add_argument('--saveto', required=False, default='out', help='output dir')
parser.add_argument('--blocks', type=int, default=6, dest='blocks', help='Number of blocks (iterations)')
parser.add_argument('--bs', type=int, default=1, help='Batch size')
parser.add_argument('--height', required=False, default=320, type=int)
parser.add_argument('--width', required=False, default=180, type=int)
parser.add_argument('--net', required=False, type=str, default='GD', help='GD: Unrolled Gradiant Descent; NN: Neumann Network')
parser.add_argument('--load', dest='load', type=int, required=True, default=-1, help='Load model from a .pth file by epoch #')
parser.add_argument('--state', required=False, type=str, default='train')
parser.add_argument('--class_name', required=False, type=str, default='C')
args = parser.parse_args()

def batch_test(net):
    for i, data in enumerate(m.dataloader):
        y = data[0].to(device)
        results = m.test(y)
        print(f'Saving images for batch {i}')
        for j in range(y.size()[0]):
            vutils.save_image(results[j,0], f'{args.saveto}/{class_name}/{fnames[i*args.bs+j]}', normalize=True)  # to 0~255

            
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if args.datadir is None:
    raise ValueError("`datadir` parameter is required for dataset")

dataset = dset.ImageFolder(root=args.datadir,
                            transform=transforms.Compose([
                                transforms.Resize((args.height,args.width)),
                                transforms.Grayscale(), # 1 channel
                                transforms.ToTensor(),
                            ]))

assert dataset

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

try:
    path = os.path.join(args.saveto, class_name)
    if not os.path.exists(path):
        os.makedirs(path)
        print(f'Created {path} directory')
    else:
        print(f'saving to {path}')
except OSError:
    pass


try:
    if args.net == 'GD':
        m = GradientDescentNet(args=args, dataloader=dataloader, device=device)
    elif args.net == 'NN':
        m = NeumannNet(args=args, dataloader=dataloader, device=device)
    batch_test(m)
    
except KeyboardInterrupt:
    print('Interrupted')
    try:
        sys.exit(0)
    except SystemExit:
        os._exit(0)