import torch
from torchvision.datasets import ImageFolder
import torch.nn as nn


class MultiImageFolder(ImageFolder):
    
    def __init__(self, roots, transform=None):
        self.sample_list = []
        for i in range(len(roots)):
            super(MultiImageFolder, self).__init__(root = roots[i])
            self.sample_list.append(self.samples)
        self.transform = transform

    def __getitem__(self, index):
        samples = []
        for item in self.sample_list:
            path, target = item[index]
            sample = self.loader(path)
            if self.transform is not None:
                sample = self.transform(sample)
            if self.target_transform is not None:
                target = self.target_transform(target)
            samples.append(sample)
        return samples
    
        
class DoubleImageFolder(ImageFolder):

    def __init__(self, roots, transform=None):
        super(DoubleImageFolder, self).__init__(root=roots[0])
        self.samples_0 = self.samples
        
        super(DoubleImageFolder, self).__init__(root=roots[1])
        self.samples_1 = self.samples
        
        self.transform = transform
        
        
    def __getitem__(self, index):
        path_0, target_0 = self.samples_0[index]
        path_1, target_1 = self.samples_1[index]
        
        sample_0 = self.loader(path_0)
        sample_1 = self.loader(path_1)
        if self.transform is not None:
            sample_0 = self.transform(sample_0)
        if self.target_transform is not None:
            target_0 = self.target_transform(target_0)
        if self.transform is not None:
            sample_1 = self.transform(sample_1)
        if self.target_transform is not None:
            target_1 = self.target_transform(target_1)

        return sample_0, sample_1
    
    
class TripleImageFolder(ImageFolder):

    def __init__(self, roots, transform=None):
        super(TripleImageFolder, self).__init__(root=roots[0])
        self.samples_0 = self.samples
        
        super(TripleImageFolder, self).__init__(root=roots[1])
        self.samples_1 = self.samples
        
        super(TripleImageFolder, self).__init__(root=roots[2])
        self.samples_2 = self.samples
        
        self.transform = transform
        
        
    def __getitem__(self, index):
        path_0, target_0 = self.samples_0[index]
        path_1, target_1 = self.samples_1[index]
        path_2, target_2 = self.samples_2[index]
        
        sample_0 = self.loader(path_0)
        sample_1 = self.loader(path_1)
        sample_2 = self.loader(path_2)
        
        if self.transform is not None:
            sample_0 = self.transform(sample_0)
        if self.target_transform is not None:
            target_0 = self.target_transform(target_0)
        if self.transform is not None:
            sample_1 = self.transform(sample_1)
        if self.target_transform is not None:
            target_1 = self.target_transform(target_1)
        if self.transform is not None:
            sample_2 = self.transform(sample_2)
        if self.target_transform is not None:
            target_2 = self.target_transform(target_2)

        return sample_0, sample_1, sample_2

    
def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad
                

class GANLoss(nn.Module):
    def __init__(self, gan_mode, target_real_label=.9, target_fake_label=0.0):
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
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss