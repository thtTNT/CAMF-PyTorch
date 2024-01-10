import torch
import torch.nn.functional as F


def gradient(image):
    filter = torch.tensor([[[[0.,1.,0.],[1.,-4.,1.],[0.,1.,0.]]]]).view(1,1,3,3).cuda()
    return F.conv2d(image, filter, padding=1)
