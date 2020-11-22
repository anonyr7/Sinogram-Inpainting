import torch
import torch.optim as optim
import torchvision.utils as vutils
import torch.nn as nn
import os

class Inpaint():
    def __init__(self, netG, netD, args, dataloader, device):
        self.netG = netG
        self.netD = netD
        self.dataloader = dataloader
        self.device = device
        self.outdir = args.outdir
        self.epochs = args.epochs
        self.bs = args.batchSize
        self.lr = args.lr
        self.save_cp = True
        self.start_epoch = args.load+1 if args.load>=0 else 0
        self.mask = self.gen_mask(args.width, args.ratio).to(self.device)

    def gen_mask(self, w, ratio):
        mask = torch.zeros(w)
        mask[::ratio].fill_(1)
        return mask

    def gen_x(self, y):
        return self.mask*y
    
    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
                    
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
                if not os.path.exists(os.path.join(self.outdir, 'ckpt')):
                    os.makedirs(os.path.join(self.outdir, 'ckpt'))
                    print('Created checkpoint directory')
            except OSError:
                pass
        
        optimizerG = optim.Adam(self.netG.parameters(), lr=self.lr, betas=(0.9, 0.999))
        optimizerD = optim.Adam(self.netD.parameters(), lr=self.lr, betas=(0.9, 0.999))
        
        criterion = nn.BCELoss()
        criterionL1 = torch.nn.L1Loss().to(self.device)
        criterionL2 = torch.nn.MSELoss().to(self.device)
        criterionGAN = GANLoss('vanilla').to(self.device)
#         criterionGAN = nn.BCEWithLogitsLoss().to(self.device)
#         torch.autograd.set_detect_anomaly(True)
        
        for epoch in range(self.start_epoch, self.epochs):
            print(f'D is trained {max(1, 5-epoch)} times')
            for i, data in enumerate(self.dataloader):
                for _ in range(1): #max(1, 5-epoch)
                    self.netD.zero_grad()
                    self.set_requires_grad(self.netD, True)
                    ############################
                    # Loss_D: L_D = -(log(D(y) + log(1 - D(G(x))))
                    ###########################
                    # forward
                    y = data[0].to(self.device)  # 320 x 180
                    x = self.gen_x(y).to(self.device)  # 320 x 180
                    Gx = self.netG(x)  # 320 x 180
                    
                    # train with fake
                    fake_pair = torch.cat((x, Gx), 1)
                    D_Gx = self.netD(fake_pair.detach())
                    errD_fake = criterionGAN(D_Gx, False)

                    # train with real
                    real_pair = torch.cat((x, y), 1)
                    D_y = self.netD(real_pair)
                    errD_real = criterionGAN(D_y, True)

                    # backprop
                    errD = (errD_real + errD_fake) * 0.5
                    errD.backward()
                    optimizerD.step()
                
                self.netG.zero_grad()
                self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
                ############################
                # Loss_G_GAN: L_G = -log(D(G(x))  # Fake the D
                ###########################
                D_Gx = self.netD(fake_pair)
                errG_GAN = criterionGAN(D_Gx, True)
                
                ############################
                # Loss_G_C: L_C = ||y - G(x)||_1
                ###########################
                errG_C = criterionL2(Gx, y)*50
#                 errG_CM = criterionL2((1-self.mask)*Gx, (1-self.mask)*y) * 50
                
                # backprop
                errG = errG_GAN + errG_C
                errG.backward()
                optimizerG.step()
                
                if i % 100 == 0:
                    print(f'[{epoch}/{self.epochs}][{i}/{len(self.dataloader)}] LossD_real: {errD_real.item():.4f} LossD_fake: {errD_fake.item():.4f} LossG_GAN: {errG_GAN.item():.4f} LossG_C: {errG_C.item():.4f}')
                
            # Log
            print(f'[{epoch}/{self.epochs}][{i}/{len(self.dataloader)}] LossD: {errD.item():.4f} LossG: {errG.item():.4f}')
            if self.save_cp:
                torch.save(self.netG.state_dict(), f'{self.outdir}/ckpt/G_epoch{epoch}.pth')
                torch.save(self.netD.state_dict(), f'{self.outdir}/ckpt/D_epoch{epoch}.pth')
            vutils.save_image(Gx.detach(), '%s/impainted_samples_epoch_%03d.png' % (self.outdir, epoch), normalize=True)

            
class GANLoss(nn.Module):
    """Define different GAN objectives.
    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.
        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image
        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss() # sigmoid inside
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.
        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.
        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss