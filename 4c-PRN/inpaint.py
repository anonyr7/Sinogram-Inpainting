import torch
import torch.optim as optim
import torchvision.utils as vutils
import torch.nn as nn
import os
from utils import *


class Inpaint():
    def __init__(self, netG, netD, args, dataloader, device):
        self.netG = netG
        self.netD = netD
        self.dataloader = dataloader
        self.device = device
        self.args = args
        self.epochs = args.epochs
        self.bs = args.batchSize
        self.lr = args.lr
        self.save_cp = True
        self.start_epoch = args.load+1 if args.load>=0 else 0
        
        self.criterionL1 = torch.nn.L1Loss().to(self.device)
        self.criterionL2 = torch.nn.MSELoss().to(self.device)
        self.criterionGAN = GANLoss('vanilla').to(self.device)
        
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=self.lr, betas=(0.5, 0.999))
        self.optimizerD = optim.Adam(self.netD.parameters(), lr=self.lr, betas=(0.5, 0.999))
        
        
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
    
    
    def train(self):
        print(f'''Starting training:
            Epochs:          {self.epochs}
            Batch size:      {self.bs}
            Learning rate:   {self.lr}
            Checkpoints:     {self.save_cp}
            Device:          {self.device.type}
        ''')
        
        if self.save_cp:
            try:
                if not os.path.exists(os.path.join(self.args.outdir, 'ckpt')):
                    os.makedirs(os.path.join(self.args.outdir, 'ckpt'))
                    print('Created checkpoint directory')
            except OSError:
                pass
        

        for epoch in range(self.start_epoch, self.epochs):
            for i, data in enumerate(self.dataloader):
                self.netD.zero_grad()
                set_requires_grad(self.netD, True)
                
                y = data[0].to(self.device)  # GT, 320 x 320
                if self.args.input_channel == 2:
                    x1 = data[1].to(self.device)  # input, 320 x 320
                    x2 = data[2].to(self.device)  # input, 320 x 320
                    x = torch.cat((x1,x2),1)
                elif self.args.input_channel == 4:
                    x1 = data[1].to(self.device)  # input, 320 x 320
                    x2 = data[2].to(self.device)  # input, 320 x 320
                    x3 = data[3].to(self.device)  # input, 320 x 320
                    x4 = data[4].to(self.device)  # input, 320 x 320
                    x = torch.cat((x1,x2,x3,x4),1)
                else:
                    x = data[1]
                
                # forward
                Gx = self.netG(x)  # 320 x 320
                
                ############################
                # Loss_D: L_D = -(log(D(y) + log(1 - D(G(x))))
                ###########################
                # train with fake
                D_Gx = self.netD(Gx.detach())[-1]
                errD_fake = self.criterionGAN(D_Gx, False)

                # train with real
                D_y = self.netD(y)[-1]
                errD_real = self.criterionGAN(D_y, True)

                # backprop
                errD = (errD_real + errD_fake) * 0.5
                errD.backward()
                self.optimizerD.step()
                
                self.netG.zero_grad()
                set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
                ############################
                # Loss_G_GAN: L_G = -log(D(G(x))  # Fake the D
                ###########################
                Gx_features = self.netD(Gx)
                errG_GAN = self.criterionGAN(Gx_features[-1], True)
                
                ############################
                # Loss_G_C: L_C = ||y - G(x)||_1
                ###########################
                errG_C = self.criterionL1(Gx, y)*50
                
                ############################
                # Loss_G_P: Perceptual feature loss
                # Loss_G_DP: Discriminator perceptual feature loss
                ###########################
                y_features = self.netD(y)
                errG_DP = self.criterionDP(Gx_features[:-1],y_features[:-1])*20

                errG = errG_C + errG_GAN + errG_DP
                errG.backward()
                self.optimizerG.step()
                
                if i % 100 == 0:
                    print(f'[{epoch}/{self.epochs}][{i}/{len(self.dataloader)}] LossD_real: {errD_real.item():.4f} LossD_fake: {errD_fake.item():.4f} LossG_GAN: {errG_GAN.item():.4f} LossG_C: {errG_C.item():.4f} LossG_DP: {errG_DP.item():.4f}')

            # Log
            print(f'[{epoch}/{self.epochs}][{i}/{len(self.dataloader)}] LossD: {errD.item():.4f} LossG: {errG.item():.4f}')
            if self.save_cp:
                torch.save(self.netG.state_dict(), f'{self.args.outdir}/ckpt/G_epoch{epoch}.pth')
                torch.save(self.netD.state_dict(), f'{self.args.outdir}/ckpt/D_epoch{epoch}.pth')
            vutils.save_image(Gx.detach(), '%s/impainted_samples_epoch_%03d.png' % (self.args.outdir, epoch), normalize=True)
