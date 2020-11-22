import torch
import torch.nn as nn


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

def gen_hole_area(hole_size, mask_size):
    """
    * inputs:
        - hole_size (sequence, required)
                A sequence of length 2 (W, H) is assumed.
                (W, H) is the size of hole area.
        - mask_size (sequence, required)
                A sequence of length 2 (W, H) is assumed.
                (W, H) is the size of input mask.
    * returns:
            A sequence used for the input argument 'hole_area' for function 'gen_input_mask'.
    """
    mask_w, mask_h = mask_size
    harea_w, harea_h = hole_size
    offset_x = torch.randint(mask_w-harea_w, (1,))
    offset_y = torch.randint(mask_h-harea_h, (1,))
    return ((offset_x, offset_y), (harea_w, harea_h))

def crop(x, area):
    """
    * inputs:
        - x (torch.Tensor, required)
                A torch tensor of shape (N, C, H, W) is assumed.
        - area (sequence, required)
                A sequence of length 2 ((X, Y), (W, H)) is assumed.
                sequence[0] (X, Y) is the left corner of an area to be cropped.
                sequence[1] (W, H) is its width and height.
    * returns:
            A torch tensor of shape (N, C, H, W) cropped in the specified area.
    """
    xmin, ymin = area[0]
    w, h = area[1]
    return x[:, :, ymin : ymin + h, xmin : xmin + w]