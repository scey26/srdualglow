from tqdm import tqdm
import numpy as np
from PIL import Image
from math import log, sqrt, pi

import argparse

import torch
from torch import nn, optim
from torch.autograd import Variable, grad
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
from dataset import ImageDataset
from torch.utils.data import DataLoader
from model import Glow, prep_conds
from cond_glow import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description="Glow trainer")
parser.add_argument("--batch", default=16, type=int, help="batch size")
parser.add_argument("--iter", default=200000, type=int, help="maximum iterations")
parser.add_argument("--total_epoch", default=100, type=int, help="maximum epochs")
parser.add_argument(
    "--n_flow", default=16, type=int, help="number of flows in each block"
)
parser.add_argument("--n_block", default=2, type=int, help="number of blocks")
parser.add_argument(
    "--no_lu",
    action="store_true",
    help="use plain convolution instead of LU decomposed version",
)
parser.add_argument(
    "--affine", action="store_true", help="use affine coupling instead of additive"
)
parser.add_argument("--n_bits", default=5, type=int, help="number of bits")
parser.add_argument("--lr", default=1e-4, type=float, help="learning rate")
parser.add_argument("--img_size", default=64, type=int, help="image size")
parser.add_argument("--scale", default=2, type=int, help="scale size")
parser.add_argument("--temp", default=0.7, type=float, help="temperature of sampling")
parser.add_argument("--n_sample", default=20, type=int, help="number of samples")
parser.add_argument("path", metavar="PATH", type=str, help="Path to image directory")


def sample_data(path, batch_size, image_size, scale=1):
    transform = transforms.Compose(
        [
            transforms.Resize(image_size // scale),
            transforms.CenterCrop(image_size // scale),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    dataset = datasets.ImageFolder(path, transform=transform)
    # dataset = datasets.CelebA(root='./dataset', split='train', transform=transform, download=True)
    loader = DataLoader(dataset, shuffle=False, batch_size=batch_size, num_workers=4)
    loader = iter(loader)

    while True:
        try:
            yield next(loader)

        except StopIteration:
            loader = DataLoader(
                dataset, shuffle=True, batch_size=batch_size, num_workers=4
            )
            loader = iter(loader)
            yield next(loader)


def calc_z_shapes(n_channel, input_size, n_flow, n_block):
    z_shapes = []

    for i in range(n_block - 1):
        input_size //= 2
        n_channel *= 2

        z_shapes.append((n_channel, input_size, input_size))

    input_size //= 2
    z_shapes.append((n_channel * 4, input_size, input_size))

    return z_shapes


def calc_loss(log_p, logdet, image_size, n_bins):
    # log_p = calc_log_p([z_list])
    n_pixel = image_size * image_size * 3

    loss = -log(n_bins) * n_pixel
    loss = loss + logdet + log_p

    return (
        (-loss / (log(2) * n_pixel)).mean(),
        (log_p / (log(2) * n_pixel)).mean(),
        (logdet / (log(2) * n_pixel)).mean(),
    )


def train(args, model_left, model_right, optimizer_left, optimizer_right):
    lr_dataset = iter(sample_data(args.path, args.batch, args.img_size, scale=args.scale))
    hr_dataset = iter(sample_data(args.path, args.batch, args.img_size, scale=1))
    n_bins = 2.0 ** args.n_bits

    z_sample_left = []
    z_shapes_left = calc_z_shapes(3, args.img_size // args.scale, args.n_flow, args.n_block)
    for z in z_shapes_left:
        z_new = torch.randn(args.n_sample, *z) * args.temp
        z_sample_left.append(z_new.to(device))

    z_sample_right = []
    z_shapes_right = calc_z_shapes(3, args.img_size, args.n_flow, args.n_block)
    for z in z_shapes_right:
        z_new = torch.randn(args.n_sample, *z) * args.temp
        z_sample_right.append(z_new.to(device))

    with tqdm(range(args.iter)) as pbar:
        for i in pbar:
            img_lr, _ = next(lr_dataset)
            img_lr = img_lr.to(device)
            img_lr = img_lr * 255

            img_hr, _ = next(hr_dataset)
            img_hr = img_hr.to(device)
            img_hr = img_hr * 255


            if args.n_bits < 8:
                img_lr = torch.floor(img_lr / 2 ** (8 - args.n_bits))

            img_lr = img_lr / n_bins - 0.5

            if args.n_bits < 8:
                img_hr = torch.floor(img_hr / 2 ** (8 - args.n_bits))

            img_hr = img_hr / n_bins - 0.5

            if i == 0:
                with torch.no_grad():
                    # lr_log_p, lr_logdet, _ = model_left.module(
                    #     img_lr + torch.rand_like(img_lr) / n_bins
                    # )

                    left_glow_out = model_left.module(
                        img_lr + torch.rand_like(img_lr) / n_bins
                    )

                    lr_log_p, lr_logdet = left_glow_out['log_p_sum'], left_glow_out['log_det']

                    conditions = prep_conds(left_glow_out, direction='forward')

                    # hr_log_p, hr_logdet, _ = model_right.module(
                    #     img_hr + torch.rand_like(img_hr) / n_bins, conditions
                    # )

                    right_glow_out = model_right.module(
                        img_hr + torch.rand_like(img_hr) / n_bins, conditions
                    )

                    hr_log_p, hr_logdet = right_glow_out['log_p_sum'], right_glow_out['log_det']

                    continue

            else:
                # lr_log_p, lr_logdet, _ = model_left(img_lr + torch.rand_like(img_lr) / n_bins)
                left_glow_out = model_left(
                        img_lr + torch.rand_like(img_lr) / n_bins
                    )
                lr_log_p, lr_logdet = left_glow_out['log_p_sum'], left_glow_out['log_det']
                conditions = prep_conds(left_glow_out, direction='forward')
                # hr_log_p, hr_logdet, _ = model_right(img_hr + torch.rand_like(img_hr) / n_bins, conditions)
                right_glow_out = model_right(img_hr + torch.rand_like(img_hr) / n_bins, conditions)
                hr_log_p, hr_logdet = right_glow_out['log_p_sum'], right_glow_out['log_det']

            lr_logdet = lr_logdet.mean()
            hr_logdet = hr_logdet.mean()

            lr_loss, lr_log_p, lr_log_det = calc_loss(lr_log_p, lr_logdet, args.img_size // args.scale, n_bins)
            model_left.zero_grad()
            lr_loss.backward()
            # warmup_lr = args.lr * min(1, i * batch_size / (50000 * 10))
            warmup_lr = args.lr
            optimizer_left.param_groups[0]["lr"] = warmup_lr
            optimizer_left.step()


            
            hr_loss, hr_log_p, hr_log_det = calc_loss(hr_log_p, hr_logdet, args.img_size, n_bins)
            model_right.zero_grad()
            hr_loss.backward()
            # warmup_lr = args.lr * min(1, i * batch_size / (50000 * 10))
            warmup_lr = args.lr
            optimizer_right.param_groups[0]["lr"] = warmup_lr
            optimizer_right.step()


            pbar.set_description(
                f"lr_Loss: {lr_loss.item():.5f}; lr_logP: {lr_log_p.item():.5f}; lr_logdet: {lr_log_det.item():.5f}; lr: {warmup_lr:.7f},"
            )
        

            if i % 1 == 0:
                with torch.no_grad():
                    utils.save_image(
                        model_single_left.reverse(z_sample_left).cpu().data,
                        f"sample/{str(i + 1).zfill(6)}_left.png",
                        normalize=True,
                        nrow=10,
                        range=(-0.5, 0.5),
                    )

                    left_glow_out = model_left(
                        img_lr + torch.rand_like(img_lr) / n_bins
                    )

                    conditions = prep_conds(left_glow_out, direction='reverse')

                    utils.save_image(
                        model_single_right.reverse(z_sample_right).cpu().data,
                        f"sample/{str(i + 1).zfill(6)}_right.png",
                        normalize=True,
                        nrow=10,
                        range=(-0.5, 0.5),
                    )


            if i % 1 == 0:
                torch.save(
                    model_left.state_dict(), f"checkpoint/model_{str(i + 1).zfill(6)}.pt"
                )
                torch.save(
                    optimizer_left.state_dict(), f"checkpoint/optim_{str(i + 1).zfill(6)}.pt"
                )

                torch.save(
                    model_left.state_dict(), f"checkpoint/model_{str(i + 1).zfill(6)}.pt"
                )
                torch.save(
                    optimizer_left.state_dict(), f"checkpoint/optim_{str(i + 1).zfill(6)}.pt"
                )


def calc_z_shapes2(n_channel, image_size, n_block, split_type):
    # calculates shapes of z's after SPLIT operation (after Block operations) - e.g. channels: 6, 12, 24, 96
    z_shapes = []
    for i in range(n_block - 1):
        image_size = (image_size[0] // 2, image_size[1] // 2)
        n_channel = n_channel * 2 if split_type == 'regular' else 9  # now only supports split_sections [3, 9]

        shape = (n_channel, *image_size)
        z_shapes.append(shape)

    # for the very last block where we have no split operation
    image_size = (image_size[0] // 2, image_size[1] // 2)
    shape = (n_channel * 4, *image_size) if split_type == 'regular' else (12, *image_size)
    z_shapes.append(shape)
    return z_shapes

def calc_inp_shapes(n_channels, image_size, n_blocks, split_type):
    # calculates z shapes (inputs) after SQUEEZE operation (before Block operations) - e.g. channels: 12, 24, 48, 96
    z_shapes = calc_z_shapes2(n_channels, image_size, n_blocks, split_type)
    input_shapes = []
    for i in range(len(z_shapes)):
        if i < len(z_shapes) - 1:
            channels = z_shapes[i][0] * 2 if split_type == 'regular' else 12  # now only supports split_sections [3, 9]
            input_shapes.append((channels, z_shapes[i][1], z_shapes[i][2]))
        else:
            input_shapes.append((z_shapes[i][0], z_shapes[i][1], z_shapes[i][2]))
    return input_shapes


def calc_cond_shapes(n_channels, image_size, n_blocks, split_type, condition):
    # computes additional channels dimensions based on additional conditions: left input + condition
    input_shapes = calc_inp_shapes(n_channels, image_size, n_blocks, split_type)
    cond_shapes = []
    for block_idx in range(len(input_shapes)):
        shape = [input_shapes[block_idx][0], input_shapes[block_idx][1], input_shapes[block_idx][2]]  # from left glow
        if 'b_maps' in condition:
            shape[0] += 3   # down-sampled image with 3 channels
        cond_shapes.append(tuple(shape))
    return cond_shapes

if __name__ == "__main__":
    args = parser.parse_args()
    print(args)

    left_configs = {'all_conditional': False, 'split_type': 'regular', 'do_lu': False, 'grad_checkpoint': False}  # default
    right_configs = {'all_conditional': True, 'split_type': 'regular', 'do_lu': False, 'condition': 'left', 'grad_checkpoint': False}  # default condition from left glow

    input_shapes = calc_inp_shapes(3, (args.img_size, args.img_size), args.n_block, left_configs['split_type'])
    output_shapes = calc_inp_shapes(3, (args.img_size, args.img_size), args.n_block, left_configs['split_type'])
    cond_shapes = calc_cond_shapes(3, (args.img_size, args.img_size), args.n_block, right_configs['split_type'], right_configs['condition'])  # shape (C, H, W)


    # left side
    model_single_left = Cond_Glow(n_blocks=2, n_flows=[8, 8], input_shapes=input_shapes, cond_shapes=None, configs=left_configs)
    model_left = nn.DataParallel(model_single_left)
    # model = model_single
    model_left = model_left.to(device) 

    optimizer_left = optim.Adam(model_left.parameters(), lr=args.lr)
 
    # right side
    model_single_right = Cond_Glow(n_blocks=2, n_flows=[8, 8], input_shapes=output_shapes,cond_shapes=cond_shapes,configs=right_configs)
    model_right = nn.DataParallel(model_single_right)
    # model = model_single
    model_right = model_right.to(device)

    optimizer_right = optim.Adam(model_right.parameters(), lr=args.lr)

    train(args, model_left, model_right, optimizer_left, optimizer_right)
