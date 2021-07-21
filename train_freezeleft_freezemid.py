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
parser.add_argument("--batch", default=4, type=int, help="batch size")
parser.add_argument("--iter", default=200000, type=int, help="maximum iterations")
parser.add_argument("--n_flow", default=32, type=int, help="number of flows in each block")
parser.add_argument("--n_block", default=4, type=int, help="number of blocks")
parser.add_argument("--no_lu", action="store_true", help="use plain convolution instead of LU decomposed version")
parser.add_argument("--affine", action="store_false", help="use affine coupling instead of additive")
parser.add_argument("--n_bits", default=5, type=int, help="number of bits")
parser.add_argument("--lr", default=1e-4, type=float, help="learning rate")
parser.add_argument("--img_size", default=128, type=int, help="image size")
parser.add_argument("--scale", default=2, type=int, help="SR scale")
parser.add_argument("--temp", default=0.7, type=float, help="temperature of sampling")
parser.add_argument("--n_sample", default=4, type=int, help="number of samples") # it should be same with batch size
parser.add_argument("path", metavar="PATH", type=str, help="Path to image directory")
parser.add_argument("--left_glow_params", type=str)
parser.add_argument("--mid_glow_params", type=str)
parser.add_argument("--save_folder", type=str)

def sample_data(path, batch_size, image_size):
    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    dataset = datasets.ImageFolder(path, transform=transform)
    # print(dataset)
    # dataset = datasets.CelebA(root='./dataset', split='train', transform=transform, download=True)
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


def train(args, model_left, model_right, optimizer):
    dataset_lr = iter(sample_data(args.path, args.batch, args.img_size // args.scale))
    dataset_mr = iter(sample_data(args.path, args.batch, args.img_size // (args.scale // 2)))
    dataset_hr = iter(sample_data(args.path, args.batch, args.img_size))
    n_bins = 2.0 ** args.n_bits

    z_sample_mr = []
    z_shapes_mr = calc_z_shapes(3, args.img_size // 2, args.n_flow, args.n_block)
    for z in z_shapes_mr:
        z_new = torch.randn(args.batch, *z) * args.temp
        z_sample_mr.append(z_new.to(device))


    z_sample_hr = []
    z_shapes_hr = calc_z_shapes(3, args.img_size, args.n_flow, args.n_block)
    for z in z_shapes_hr:
        z_new = torch.randn(args.batch, *z) * args.temp
        z_sample_hr.append(z_new.to(device))

    with tqdm(range(args.iter)) as pbar:
        for i in pbar:
            image_lr, _ = next(dataset_lr)  # 32
            image_lr = image_lr.to(device)

            image_lr = image_lr * 255

            if args.n_bits < 8:
                image_lr = torch.floor(image_lr / 2 ** (8 - args.n_bits))

            image_lr = image_lr / n_bins - 0.5

            image_mr, _ = next(dataset_mr)
            image_mr = image_mr.to(device)

            image_mr = image_mr * 255

            if args.n_bits < 8:
                image_mr = torch.floor(image_mr / 2 ** (8 - args.n_bits))

            image_mr = image_mr / n_bins - 0.5

            image_hr, _ = next(dataset_hr)
            image_hr = image_hr.to(device)

            image_hr = image_hr * 255

            if args.n_bits < 8:
                image_hr = torch.floor(image_hr / 2 ** (8 - args.n_bits))

            image_hr = image_hr / n_bins - 0.5

            if i == 0:
                with torch.no_grad():
                    #  perform left glow forward
                    left_glow_out = model_left.module(
                        image_lr + torch.rand_like(image_lr) / n_bins
                    )

                    # perform right glow forward
                    mid_glow_out = model_mid.module(image_mr + torch.rand_like(image_mr) / n_bins, left_glow_out)

                    right_glow_out = model_right.module(image_hr + torch.rand_like(image_hr) / n_bins, mid_glow_out)

                    continue

            else:
                #  perform left glow forward
                left_glow_out = model_left(
                    image_lr + torch.rand_like(image_lr) / n_bins
                )

                # perform right glow forward
                mid_glow_out = model_mid(image_mr + torch.rand_like(image_mr) / n_bins, left_glow_out)

                right_glow_out = model_right(image_hr + torch.rand_like(image_hr) / n_bins, mid_glow_out)

                # extract right outputs
                log_p_hr, logdet_hr = right_glow_out['log_p_sum'], right_glow_out['log_det']


            warmup_lr = args.lr

            logdet_hr = logdet_hr.mean()

            loss_hr, log_p_hr, log_det_hr = calc_loss(log_p_hr, logdet_hr, args.img_size, n_bins)

            optimizer.zero_grad()
            loss =  loss_hr
            loss.backward()
            optimizer.step()

            pbar.set_description(
                f"Loss: {loss_hr.item():.5f}; logP: {log_p_hr.item():.5f}; logdet: {log_det_hr.item():.5f}; lr: {warmup_lr:.7f}"
            )

            if i % 100 == 0:
                with torch.no_grad():
                    utils.save_image(image_lr,
                        f"sample/{args.save_folder}/gt_lr_{str(i + 1).zfill(6)}.png",
                        normalize=True,
                        value_range=(-0.5, 0.5),
                    )

                    utils.save_image(image_hr,
                        f"sample/{args.save_folder}/gt_hr_{str(i + 1).zfill(6)}.png",
                        normalize=True,
                        value_range=(-0.5, 0.5),
                    )

                    utils.save_image(image_mr,
                        f"sample/{args.save_folder}/gt_mr_{str(i + 1).zfill(6)}.png",
                        normalize=True,
                        value_range=(-0.5, 0.5),
                    )

                    utils.save_image(
                        model_single_left.reverse(left_glow_out['z_outs'], reconstruct=True).cpu().data,
                        f"sample/{args.save_folder}/gen_lr_{str(i + 1).zfill(6)}.png",
                        normalize=True,
                        nrow=10,
                        value_range=(-0.5, 0.5),
                    )

                    utils.save_image(
                        model_single_mid.reverse(mid_glow_out['z_outs'], reconstruct=True, left_glow_out=left_glow_out).cpu().data,
                        f"sample/{args.save_folder}/gen_mr_{str(i + 1).zfill(6)}.png",
                        normalize=True,
                        nrow=10,
                        value_range=(-0.5, 0.5),
                    )

                    utils.save_image(
                        model_single_right.reverse(right_glow_out['z_outs'], reconstruct=True, left_glow_out=mid_glow_out).cpu().data,
                        f"sample/{args.save_folder}/gen_hr_{str(i + 1).zfill(6)}.png",
                        normalize=True,
                        nrow=10,
                        value_range=(-0.5, 0.5),
                    )

                    utils.save_image(
                        model_single_right.reverse(z_sample_hr, reconstruct=False, left_glow_out=mid_glow_out).cpu().data,
                        f"sample/{args.save_folder}/gen_hr_randz_{str(i + 1).zfill(6)}.png",
                        normalize=True,
                        nrow=10,
                        value_range=(-0.5, 0.5),
                    )

                    gen_mr_randz = model_single_mid.reverse(z_sample_mr, reconstruct=False, left_glow_out=left_glow_out)
                    mid_glow_out_randz = model_mid(gen_mr_randz + torch.rand_like(gen_mr_randz) / n_bins, left_glow_out)

                    utils.save_image(
                        model_single_right.reverse(z_sample_hr, reconstruct=False, left_glow_out=mid_glow_out_randz).cpu().data,
                        f"sample/{args.save_folder}/gen_hr_randz_randz_{str(i + 1).zfill(6)}.png",
                        normalize=True,
                        nrow=10,
                        value_range=(-0.5, 0.5),
                    )

            if i % 10000 == 0:
                torch.save(
                    optimizer.state_dict(), f"checkpoint/{args.save_folder}/optim_{str(i + 1).zfill(6)}.pt"
                )
                torch.save(
                    model_right.state_dict(), f"checkpoint/{args.save_folder}/model_hr_{str(i + 1).zfill(6)}.pt"
                )


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    os.makedirs(f"checkpoint/{args.save_folder}",exist_ok = True)
    os.makedirs(f"sample/{args.save_folder}",exist_ok = True)

    input_shapes = calc_inp_shapes(3, [args.img_size , args.img_size], args.n_block)
    input_shapes_mid = calc_inp_shapes(3, [args.img_size // 2, args.img_size // 2], args.n_block)

    cond_shapes = calc_cond_shapes(3, [args.img_size, args.img_size], args.n_block)
    cond_shapes_mid = calc_cond_shapes(3, [args.img_size // 2, args.img_size // 2], args.n_block)

    model_single_left = Glow(
        3, args.n_flow, args.n_block, affine=False, conv_lu=not args.no_lu
    )
    model_left = nn.DataParallel(model_single_left)
    model_left = model_left.to(device)
    model_left.load_state_dict(torch.load(args.left_glow_params))
    model_left.eval()

    model_single_mid = Cond_Glow(
        3, args.n_flow, args.n_block, input_shapes_mid, cond_shapes_mid, affine=args.affine, conv_lu=not args.no_lu
    )
    model_mid = nn.DataParallel(model_single_mid)
    model_mid = model_mid.to(device)
    model_mid.load_state_dict(torch.load(args.mid_glow_params))
    model_mid.eval()

    model_single_right = Cond_Glow(
        3, args.n_flow, args.n_block, input_shapes, cond_shapes, affine=args.affine, conv_lu=not args.no_lu
    )
    model_right = nn.DataParallel(model_single_right)
    model_right = model_right.to(device)
    
    optimizer = optim.Adam([{"params": model_right.parameters(), "lr": args.lr}])

    train(args, model_left, model_right, optimizer)
