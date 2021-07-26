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
parser.add_argument("--iter", default=100, type=int, help="maximum iterations")
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
parser.add_argument("--left_glow_params", type=str)
parser.add_argument("--save_folder", type=str)


def sample_data(path, batch_size, image_size):
    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            # transforms.RandomHorizontalFlip(), # For pairing LR-HR, future work
            transforms.ToTensor(),
        ]
    )

    dataset = datasets.ImageFolder(path, transform=transform)
    loader = DataLoader(dataset, shuffle=False, batch_size=batch_size, num_workers=4)
    loader = iter(loader)

    while True:
        try:
            yield next(loader)

        except StopIteration:
            loader = DataLoader(
                dataset, shuffle=False, batch_size=batch_size, num_workers=4
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


def test(args, model_left):
    dataset_lr = iter(sample_data(args.path, args.batch, args.img_size // args.scale))
    n_bins = 2.0 ** args.n_bits
    z_shapes_lr = calc_z_shapes(3, args.img_size // args.scale, args.n_flow, args.n_block)

    with torch.no_grad():
        with tqdm(range(args.iter)) as pbar:
            for i in pbar:
                image_lr, _ = next(dataset_lr)
                image_lr = image_lr.to(device)

                image_lr = image_lr * 255

                if args.n_bits < 8:
                    image_lr = torch.floor(image_lr / 2 ** (8 - args.n_bits))

                image_lr = image_lr / n_bins - 0.5

                #  perform left glow forward
                left_glow_out = model_left(
                    image_lr + torch.rand_like(image_lr) / n_bins
                )

                # also generate random z for left glow backward
                z_sample_lr = []
                for z in z_shapes_lr:
                    z_new = torch.randn(args.batch, *z) * args.temp
                    z_sample_lr.append(z_new.to(device))

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


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)

    os.makedirs(f"sample/{args.save_folder}", exist_ok=True)

    input_shapes = calc_inp_shapes(3, [args.img_size // args.scale, args.img_size // args.scale], args.n_block)

    cond_shapes = calc_cond_shapes(3, [args.img_size, args.img_size], args.n_block)

    model_single_left = Glow(
        3, args.n_flow, args.n_block, affine=args.affine, conv_lu=not args.no_lu
    )
    model_left = nn.DataParallel(model_single_left)

    model_left = model_left.to(device)
    model_left = model_left.to(device)
    model_left.load_state_dict(torch.load(args.left_glow_params))
    model_left.eval()

    test(args, model_left)
