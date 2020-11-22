import torch
import torch.optim as optim
import torchvision.utils as vutils
import torch.nn as nn
import os
import math
from utils import *
from model import *
from torch_radon import Radon
import numpy as np
from csv import writer
import time


class Inpaint():
    def __init__(self, net, args, dataloader, device):
        self.netG = net[0]
        self.netDG = net[1]
        self.netDL = net[2]
        if args.mode == 'vgg':
            self.netLoss = net[3]
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=args.lr, betas=(0.5, 0.999))
        self.optimizerDG = optim.Adam(self.netDG.parameters(), lr=args.lr, betas=(0.5, 0.999))
        self.optimizerDL = optim.Adam(self.netDL.parameters(), lr=args.lr, betas=(0.5, 0.999))
        
        self.dataloader = dataloader
        self.device = device
        self.args = args
        self.save_cp = True
        self.start_epoch = args.load+1 if args.load>=0 else 0
        self.mask = self.gen_mask().to(self.device)
        
        self.criterionL1 = torch.nn.L1Loss().to(self.device)
        self.criterionL2 = torch.nn.MSELoss().to(self.device)
        self.criterionGAN = GANLoss('vanilla').to(self.device)
        
        err_list = ["errDG", "errDL", 
                    "errGG_GAN", "errGG_C", "errGG_F", "errGG_P",
                    "errGL_GAN", "errGL_C", "errGL_F", "errGL_P"]
        self.err = dict.fromkeys(err_list, None) 
                    
        if self.save_cp:
            try:
                if not os.path.exists(os.path.join(args.outdir, 'ckpt')):
                    os.makedirs(os.path.join(args.outdir, 'ckpt'))
                    print('Created checkpoint directory')
                if args.load < 0:  # New log file
                    with open(os.path.join(args.outdir, args.log_fn+'.csv'), 'w', newline='') as f:
                        csvwriter = writer(f)
                        csvwriter.writerow(["epoch", "runtime"] + err_list)
            except OSError:
                pass
        
        angles = np.linspace(0, np.pi, 180, endpoint=False)
        self.radon = Radon(args.height, angles, clip_to_circle=True)

        
    def gen_mask(self):
        mask = torch.zeros(180)
        mask[::8].fill_(1)  # 180/23
        if self.args.twoends:
            mask = torch.cat((mask[-6:], mask, mask[:6]), 0) # 192/25
        return mask
            
    
    def gen_sparse(self, y):
        return y[:,:,:,self.mask==1]
    
    
    def append_twoends(self, y):
        front = torch.flip(y[:,:,:,:6], [2])
        back = torch.flip(y[:,:,:,-6:], [2])
        return torch.cat((back, y, front), 3)
    
    
    def ramp_module(self, sinogram):
        '''
            Sinogram has dimension: bs x c x height x angle. 
            Ramp is 1D but angle number affects normalization for filter_sinogram. Use with caution.
        '''
        normalized_sinogram = normalize(sinogram, rto=(0,1))
        if sinogram.size()[2] == self.args.height:
            filtered_sinogram = self.radon.filter_sinogram(normalized_sinogram.permute(0,1,3,2)).permute(0,1,3,2)  # 320 x 192
        else:
            print('sinogram dimension wrong for filter!')
        return normalize(filtered_sinogram, rto=(-1,1))
    
    
    def criterionP(self, Gx, y):
        # calculate feature loss
        y_features = self.netLoss(y)
        Gx_features = self.netLoss(Gx)
        
        loss = 0.0
        for j in range(len(y_features)):
            loss += self.criterionL2(Gx_features[j], y_features[j][:y.shape[0]])
        return loss
    
    
    def criterionDP(self, Gx_features, y_features):
        loss = 0.0
        for j in range(len(y_features)):
            loss += self.criterionL2(Gx_features[j], y_features[j])
        return loss
        
        
    def train_D(self, Gx, y, mode):
        '''
            mode is G/L.
        '''
        if mode == 'G':
            netD = self.netDG
            optimizer = self.optimizerDG
        elif mode == 'L':
            netD = self.netDL
            optimizer = self.optimizerDL
        else:
            print('wrong mode!')
            
        netD.zero_grad()
        
        ############################
        # Loss_D: L_D = -(log(D(y) + log(1 - D(G(x))))
        ###########################
        # train with fake
        D_Gx = netD(Gx.detach())[-1]
        errD_fake = self.criterionGAN(D_Gx, False)

        # train with real
        D_y = netD(y)[-1]
        errD_real = self.criterionGAN(D_y, True)

        # backprop
        errD = (errD_real + errD_fake) * 0.5
        errD.backward()
        optimizer.step()
        self.err['errD'+mode] = errD.item()
    
    
    def train_G(self, Gx, y, filtered_Gx, filtered_y, mode):
        '''
            mode is G/L.
        '''
        if mode == 'G':
            netD = self.netDG
        elif mode == 'L':
            netD = self.netDL
        else:
            print('wrong mode!')
            
        self.netG.zero_grad()
        
        ############################
        # Loss_G_GAN: L_G = -log(D(G(x))  # Fake the D
        ###########################
        Gx_features = netD(Gx)
        errG_GAN = self.criterionGAN(Gx_features[-1], True)
        
        ############################
        # Loss_G_C: L_C = ||y - G(x)||_1
        ###########################
        errG_C = self.criterionL1(Gx, y)*50

        ############################
        # Loss_G_DP: Discriminator perceptual feature loss
        ###########################
        if self.args.mode == 'vgg':
            errG_P = self.criterionP(Gx, y)*20
        elif self.args.mode == 'DP':
            y_features = netD(y)
            errG_P = self.criterionDP(Gx_features[:-1], y_features[:-1])*20
#             errG_P = self.criterionDP(Gx_features[-2], y_features[-2])*50
        else:
            errG_P = torch.tensor(0).to(self.device)
            
        ############################
        # Loss_G_F: Ramp filtered sinogram loss
        ###########################
        errG_F = self.criterionL1(filtered_Gx, filtered_y)*50

        # backprop
        errG = errG_GAN + errG_C + errG_F + errG_P
        errG.backward()
        self.optimizerG.step()
        
        self.err['errG'+mode+'_GAN'] = errG_GAN.item()
        self.err['errG'+mode+'_C'] = errG_C.item()
        self.err['errG'+mode+'_F'] = errG_F.item()
        self.err['errG'+mode+'_P'] = errG_P.item()
    
    
    def log(self, epoch, i):
        print(f'[{epoch}/{self.args.epochs}][{i}/{len(self.dataloader)}] ' \
              f'LossDG: {self.err["errDG"]:.4f} ' \
              f'LossGG_GAN: {self.err["errGG_GAN"]:.4f} ' \
              f'LossGG_C: {self.err["errGG_C"]:.4f} ' \
              f'LossGG_F: {self.err["errGG_F"]:.4f} ' \
              f'LossGG_P: {self.err["errGG_P"]:.4f} ' \
              
              f'LossDL: {self.err["errDL"]:.4f} ' \
              f'LossGL_GAN: {self.err["errGL_GAN"]:.4f} ' \
              f'LossGL_C: {self.err["errGL_C"]:.4f} ' \
              f'LossGL_F: {self.err["errGL_F"]:.4f} ' \
              f'LossGL_P: {self.err["errGL_P"]:.4f} ' \
        )
    
    
    def log2file(self, fn, epoch, runtime):
        new_row = [epoch, runtime]+ list[self.err.values()]
        with open(fn, 'a+', newline='') as write_obj:
            csv_writer = writer(write_obj)
            csv_writer.writerow(new_row)
        
        
    def train(self):
        print(f'''Starting training:
            Epochs:          {self.args.epochs}
            Batch size:      {self.args.batchSize}
            Learning rate:   {self.args.lr}
            Checkpoints:     {self.save_cp}
            Device:          {self.device.type}
        ''')
        
        for epoch in range(self.start_epoch, self.args.epochs):
            self.D_epochs = 1 # Adjust if you want
            print('D is trained ', str(self.D_epochs), 'times in this epoch.')
            
            start = time.time()  # log start time
            for i, data in enumerate(self.dataloader):
                y = data[0].to(self.device)  # 320 x 180

                # forward
                if self.args.twoends:
                    y = self.append_twoends(y)  # 320 x 192
                
                filtered_y = self.ramp_module(y)  # 320 x 192, normalized to -1~1
                x = self.gen_sparse(y)  # 320 x 25
                
                # Train Global
                Gx = self.netG(x)
                filtered_Gx = self.ramp_module(Gx)  # 320 x 192, normalized to -1~1

                ###### Train D
                set_requires_grad(self.netDG, True)
                for _ in range(self.D_epochs):  # increase D epoch gradually. FOR DP LOSS training
                    self.train_D(Gx, y, mode='G')
                ###### Train G
                set_requires_grad(self.netDG, False)  # D requires no gradients when optimizing G
                self.train_G(Gx, y, filtered_Gx, filtered_y, mode='G')
                
                # Train Local
                Gx = self.netG(x)
                filtered_Gx = self.ramp_module(Gx)  # 320 x 192, normalized to -1~1
                patch_area = gen_hole_area((y.shape[3]//4, y.shape[2]//4), (y.shape[3], y.shape[2]))
                Gx_patch = crop(Gx, patch_area)
                y_patch = crop(y, patch_area)
                filtered_y_patch = crop(filtered_y, patch_area)
                filtered_Gx_patch = crop(filtered_Gx, patch_area)
                
                ###### Train D
                set_requires_grad(self.netDL, True)
                for _ in range(self.D_epochs):  # increase D epoch gradually. FOR DP LOSS training
                    self.train_D(Gx_patch, y_patch, mode='L')
                ###### Train G
                set_requires_grad(self.netDL, False)  # D requires no gradients when optimizing G
                self.train_G(Gx_patch, y_patch, filtered_Gx_patch, filtered_y_patch, mode='L')
                
                if i % 100 == 0:
                    self.log(epoch, i)
                    
            end = time.time()  # log end time
#             self.log2file(os.path.join(self.args.outdir, self.args.log_fn+'.csv'), epoch , str(end-start))
            
            # Log
            self.log(epoch, i)
            if self.save_cp:
                torch.save(self.netG.state_dict(), f'{self.args.outdir}/ckpt/G_epoch{epoch}.pth')
                torch.save(self.netDG.state_dict(), f'{self.args.outdir}/ckpt/DG_epoch{epoch}.pth')
                torch.save(self.netDL.state_dict(), f'{self.args.outdir}/ckpt/DL_epoch{epoch}.pth')
            vutils.save_image(Gx.detach(), '%s/impainted_samples_epoch_%03d.png' % (self.args.outdir, epoch), normalize=True)
            