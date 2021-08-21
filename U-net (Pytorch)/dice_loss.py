# Initial code from the github :
# https://github.com/milesial/Pytorch-UNet

import torch
from torch.autograd import Function
import torch.nn as nn


class IoULoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, activation=True):
        
        if activation:
            inputs = torch.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #intersection is equivalent to True Positive count
        #union is the mutually inclusive area of all labels & predictions 
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 
        
        IoU = (intersection + smooth)/(union + smooth)
                
        return 1 - IoU

def IoU_coeff(input, target):
    """IoU coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + (1 - IoULoss().forward(c[0], c[1], activation=False))

    return s / (i + 1)
