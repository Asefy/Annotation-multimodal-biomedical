# Initial code from the github :
# https://github.com/milesial/Pytorch-UNet

import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from eval import eval_net
from unet import UNet

from torch.utils.tensorboard import SummaryWriter
from utils.dataset import BasicDataset
from torch.utils.data import DataLoader

from dice_loss import IoULoss

dir_img = 'dataset_custom/'
dir_checkpoint = 'checkpoints/'

"""
Perform the prediction (binary mask for fixed image window)
of an "image" (grayscale move + mask move + grayscale fixed) concatenated

inputs :
    net :           neural network (U-net)
    device :        cuda/cpu
    epochs :        number of epoch to perform
    batch_size :    size of the batches for training
    lr :            learning rate of the Adam optimizer
    save_cp :       True|False -> True if
    img_scale :     scale at which the image should be used for training
                    (1 by default, no modification)
    loss :			"iou" or "bce"

"""
def train_net(net,
              device,
              epochs=5,
              batch_size=1,
              lr=0.001,
              save_cp=True,
              img_scale=1.,
              loss="iou"):

    torch.cuda.empty_cache()

    dataset_train = BasicDataset(dir_img, img_scale=img_scale, my_set="train", data_aug=True)
    dataset_eval = BasicDataset(dir_img, img_scale=img_scale, my_set="eval")
    print("Training samples : {}".format(len(dataset_train)))
    print("Evaluation samples : {}".format(len(dataset_eval)))
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(dataset_eval, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}_SCALE_{img_scale}')
    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {len(dataset_train)}
        Validation size: {len(dataset_eval)}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images scaling:  {img_scale}
    ''')

    # Choice of the optimizer
    #optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-8)
    #optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-6, nesterov=True)

    # Choice of the scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, gamma=0.9)

    # Choice of the loss
    if loss == "iou":
    	criterion = IoULoss()
    elif loss == "bce":
    	criterion = nn.BCEWithLogitsLoss()
    else:
    	print("Invalid loss : \"{}\"\n Use \"iou\" or \"bce\" instead".format(loss))
    

    for epoch in range(epochs):
        net.train()

        epoch_loss = 0
        with tqdm(total=len(dataset_train), desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch[0]
                true_masks = batch[1]
                assert imgs.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32
                true_masks = true_masks.to(device=device, dtype=mask_type)

                masks_pred = net(imgs)
                loss = criterion(masks_pred, true_masks)
                epoch_loss += loss.item()
                writer.add_scalar('Loss/train', loss.item(), global_step)

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                scheduler.step()

                pbar.update(imgs.shape[0])
                global_step += 1

                eval_per_epoch = 1
                if global_step % (len(dataset_train) // (eval_per_epoch * batch_size)) == 0:
                    for tag, value in net.named_parameters():
                        tag = tag.replace('.', '/')
                        writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                        writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)
                    val_score = eval_net(net, val_loader, device)
                    writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)


                    logging.info('IoU Coeff: {}'.format(val_score))
                    writer.add_scalar('IoU/test', val_score, global_step)


        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(),
                       dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')

    # save final weight (if not saved at each epochs)
    if not save_cp:
        torch.save(net.state_dict(),
                       dir_checkpoint + f'final_weight.pth')

    writer.close()


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=5,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=1,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=1.,
                        help='Downscaling factor of the images')

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    net = UNet(n_channels=3, bilinear=True)
    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    # faster convolutions, but more memory
    # cudnn.benchmark = True

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  device=device,
                  img_scale=args.scale)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
