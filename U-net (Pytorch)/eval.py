# Initial code from the github :
# https://github.com/milesial/Pytorch-UNet

import torch
import torch.nn.functional as F
from tqdm import tqdm

from dice_loss import IoU_coeff

"""
Perform the evaluation (IoU score) of a given dataset through a dataloader

inputs :
    net :       neural network (U-net)
    loader :    dataloader of the evaluation/test set
    device :    cuda/cpu
"""
def eval_net(net, loader, device):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    mask_type = torch.float32
    n_val = len(loader)  # the number of batch
    tot = 0

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs, true_masks = batch[0], batch[1]
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type)

            with torch.no_grad():
                mask_pred = net(imgs)

            pred = torch.sigmoid(mask_pred)
            pred = (pred > 0.5).float()

            tot += IoU_coeff(pred, true_masks).item()

            pbar.update()

    net.train()
    return tot / n_val
