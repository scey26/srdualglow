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

from model import Glow, Cond_Glow
from utils import *
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description="Glow trainer")
parser.add_argument("--batch", default=16, type=int, help="batch size")
parser.add_argument("--iter", default=200000, type=int, help="maximum iterations")
parser.add_argument("--n_flow", default=32, type=int, help="number of flows in each block")
parser.add_argument("--n_block", default=4, type=int, help="number of blocks")
parser.add_argument("--no_lu", action="store_true", help="use plain convolution instead of LU decomposed version")
parser.add_argument("--affine", action="store_true", help="use affine coupling instead of additive")
parser.add_argument("--n_bits", default=5, type=int, help="number of bits")
parser.add_argument("--lr", default=1e-4, type=float, help="learning rate")
parser.add_argument("--img_size", default=128, type=int, help="image size")
parser.add_argument("--scale", default=2, type=int, help="SR scale")
parser.add_argument("--temp", default=0.7, type=float, help="temperature of sampling")
parser.add_argument("--n_sample", default=1, type=int, help="number of samples")
parser.add_argument("path", metavar="PATH", type=str, help="Path to image directory")
parser.add_argument("--save_folder", type=str)




def sample_data(path, batch_size, image_size, split):
    if 'transform' not in locals() and 'dataset_train' not in locals():
        transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
        dataset_train = datasets.CelebA(root=path, split='train', transform=transform)
        dataset_valid = datasets.CelebA(root=path, split='valid', transform=transform)
        dataset_test = datasets.CelebA(root=path, split='test', transform=transform)
        print(f"Train set : {len(dataset_train)}, Valid set : {len(dataset_valid)}, 'Test set : {len(dataset_test)}")

    # dataset = datasets.ImageFolder(path, transform=transform)

    if split=='train':
        loader_train = DataLoader(dataset_train, shuffle=False, batch_size=batch_size, num_workers=4)
        loader_train = iter(loader_train)

        while True:
            try:
                yield next(loader_train)

            except StopIteration:
                loader_train = DataLoader(
                    dataset_train, shuffle=False, batch_size=batch_size, num_workers=4
                )
                loader_train = iter(loader_train)
                yield next(loader_train)

    elif split=='valid':
        loader_valid = DataLoader(dataset_valid, shuffle=False, batch_size=batch_size, num_workers=4)
        loader_valid = iter(loader_valid)

        while True:
            try:
                yield next(loader_valid)

            except StopIteration:
                loader_valid = DataLoader(
                    dataset_valid, shuffle=False, batch_size=batch_size, num_workers=4
                )
                loader_valid = iter(loader_valid)
                yield next(loader_valid)

    elif split=='test':
        loader_test = DataLoader(dataset_test, shuffle=False, batch_size=batch_size, num_workers=4)
        loader_test = iter(loader_test)

        while True:
            try:
                yield next(loader_test)

            except StopIteration:
                loader_test = DataLoader(
                    dataset_test, shuffle=False, batch_size=batch_size, num_workers=4
                )
                loader_test = iter(loader_test)
                yield next(loader_test)


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


def train(args, model_left, optimizer):
    dataset_lr = iter(sample_data(args.path, args.batch, args.img_size // args.scale, split='train'))
    n_bins = 2.0 ** args.n_bits

    z_sample_lr = []
    z_shapes_lr = calc_z_shapes(3, args.img_size // args.scale, args.n_flow, args.n_block)
    for z in z_shapes_lr:
        z_new = torch.randn(args.batch, *z) * args.temp
        z_sample_lr.append(z_new.to(device))

    with tqdm(range(args.iter)) as pbar:
        for i in pbar:
            image_lr, _ = next(dataset_lr)
            image_lr = image_lr.to(device)

            image_lr = image_lr * 255

            if args.n_bits < 8:
                image_lr = torch.floor(image_lr / 2 ** (8 - args.n_bits))

            image_lr = image_lr / n_bins - 0.5

            if i == 0:
                with torch.no_grad():
                    #  perform left glow forward
                    left_glow_out = model_left.module(
                        image_lr + torch.rand_like(image_lr) / n_bins
                    )

                    # extract left outputs
                    log_p_lr, logdet_lr = left_glow_out['log_p_sum'], left_glow_out['log_det']

                    continue

            else:
                #  perform left glow forward
                left_glow_out = model_left(
                    image_lr + torch.rand_like(image_lr) / n_bins
                )

                # extract left outputs
                log_p_lr, logdet_lr = left_glow_out['log_p_sum'], left_glow_out['log_det']

            logdet_lr = logdet_lr.mean()

            loss_lr, log_p_lr, log_det_lr = calc_loss(log_p_lr, logdet_lr, args.img_size // args.scale, n_bins)
            warmup_lr = args.lr


            optimizer.zero_grad()
            loss = loss_lr
            loss.backward()
            optimizer.step()

            pbar.set_description(
                f"Loss: {loss_lr.item():.5f} ; logP: {log_p_lr.item():.5f} ; logdet: {log_det_lr.item():.5f} ; lr: {warmup_lr:.7f}"
            )

            if i % 100 == 0:
                with torch.no_grad():
                    utils.save_image(image_lr,
                        f"sample/{args.save_folder}/gt_lr_{str(i + 1).zfill(6)}.png",
                        normalize=True,
                        range=(-0.5, 0.5),
                    )

                    utils.save_image(
                        model_single_left.reverse(left_glow_out['z_outs'], reconstruct=True).cpu().data,
                        f"sample/{args.save_folder}/gen_lr_{str(i + 1).zfill(6)}.png",
                        normalize=True,
                        nrow=10,
                        range=(-0.5, 0.5),
                    )

                    utils.save_image(
                        model_single_left.reverse(z_sample_lr).cpu().data,
                        f"sample/{args.save_folder}/gen_lr_randz_{str(i + 1).zfill(6)}.png",
                        normalize=True,
                        nrow=10,
                        range=(-0.5, 0.5),
                    )

            if i % 10000 == 0:
                torch.save(
                    model_left.state_dict(), f"checkpoint/{args.save_folder}/model_lr_{str(i + 1).zfill(6)}.pt"
                )
                torch.save(
                    optimizer.state_dict(), f"checkpoint/{args.save_folder}/optim_{str(i + 1).zfill(6)}.pt"
                )
 


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)

    os.makedirs(f"checkpoint/{args.save_folder}",exist_ok = True)
    os.makedirs(f"sample/{args.save_folder}",exist_ok = True)

    input_shapes = calc_inp_shapes(3, [args.img_size // args.scale, args.img_size // args.scale], args.n_block)

    cond_shapes = calc_cond_shapes(3, [args.img_size, args.img_size], args.n_block)

    model_single_left = Glow(
        3, args.n_flow, args.n_block, affine=args.affine, conv_lu=not args.no_lu
    )
    model_left = nn.DataParallel(model_single_left)

    model_left = model_left.to(device)

    optimizer = optim.Adam([{"params": model_left.parameters(), "lr":args.lr}])

    train(args, model_left, optimizer)
