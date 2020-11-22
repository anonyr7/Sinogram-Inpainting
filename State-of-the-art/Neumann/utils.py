import torch
import numpy as np
from torch_radon import Radon
from torch_radon.solvers import Landweber


class Operators():
    def __init__(self, image_size, n_angles, sample_ratio, device, circle=False):
        self.device = device
        self.image_size = image_size
        self.sample_ratio = sample_ratio
        self.n_angles = n_angles
        
        angles = np.linspace(0, np.pi, self.n_angles, endpoint=False)
        self.radon = Radon(self.image_size, angles, clip_to_circle=circle)
        self.radon_sparse = Radon(self.image_size, angles[::sample_ratio], clip_to_circle=circle)
        self.n_angles_sparse = len(angles[::sample_ratio])
        self.landweber = Landweber(self.radon)
        
        self.mask = torch.zeros((1,1,1,180)).to(device)
        self.mask[:,:,:,::sample_ratio].fill_(1)
        
        
    # $X^\T ()$ inverse radon
    def forward_adjoint(self, input):
        # check dimension
        if input.size()[3] == self.n_angles:
            return self.radon.backprojection(input.permute(0,1,3,2))
        elif input.size()[3] == self.n_angles_sparse:
            return self.radon_sparse.backprojection(input.permute(0,1,3,2))/self.n_angles_sparse*self.n_angles  # scale the angles
        else:
            raise Exception(f'forward_adjoint input dimension wrong! received {input.size()}.')
            
        
    # $X^\T X ()$
    def forward_gramian(self, input):
        # check dimension
        if input.size()[2] != self.image_size:
            raise Exception(f'forward_gramian input dimension wrong! received {input.size()}.')
        
        sinogram = self.radon.forward(input)
        return self.radon.backprojection(sinogram)
    

    # Corruption model: undersample sinogram by 8
    def undersample_model(self, input):
        return input[:,:,:,::self.sample_ratio]
    
    
    # Filtered Backprojection. Input siogram range = (0,1)
    def FBP(self, input):
        # check dimension
        if input.size()[2] != self.image_size or input.size()[3] != self.n_angles:
            raise Exception(f'FBP input dimension wrong! received {input.size()}.')
        filtered_sinogram = self.radon.filter_sinogram(input.permute(0,1,3,2))
        return self.radon.backprojection(filtered_sinogram)
    
    
    # estimate step size eta
    def estimate_eta(self):
        eta = self.landweber.estimate_alpha(self.image_size, self.device)
        return torch.tensor(eta, dtype=torch.float32, device=self.device)
    

def normalize(x, rfrom=None, rto=(0,1)):
    if rfrom is None:
        mean = torch.tensor([torch.min(x),]).cuda()
        std = torch.tensor([(torch.max(x)-torch.min(x)),]).cuda()
        x = x.sub(mean[None, :, None, None]).div(std[None, :, None, None]).mul(rto[1]-rto[0]).add(rto[0])
    else:
        mean = torch.tensor([rfrom[0],]).cuda()
        std = torch.tensor([rfrom[1]-rfrom[0],]).cuda()
        x = x.sub(mean[None, :, None, None]).div(std[None, :, None, None]).mul(rto[1]-rto[0]).add(rto[0])
    return x