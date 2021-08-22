# Initial code from the github :
# https://github.com/milesial/Pytorch-UNet

import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from unet import UNet
from utils.data_vis import plot_img_and_mask
from utils.dataset import BasicDataset

"""
Perform the prediction (binary mask for fixed image window)
of an "image" (grayscale move + mask move + grayscale fixed) concatenated

inputs :
    net :           neural network (U-net)
    full_img :      "image" to predict (grayscale move window + mask move window + grayscale fixed window) concatenated,
                    can be obtained by getting an item from BasicDataset of test set
    device :        cuda/cpu
    out_threshold : probability threshold to discriminate background from foreground
"""
def predict_img(net,
                full_img,
                device,
                out_threshold=0.5):
    net.eval()

    img = full_img

    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)

        probs = torch.sigmoid(output)

        probs = probs.squeeze(0)

        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(full_img.shape[1]),
                transforms.ToTensor()
            ]
        )

        probs = tf(probs.cpu())
        full_mask = probs.squeeze().cpu().numpy()

    return full_mask > out_threshold






