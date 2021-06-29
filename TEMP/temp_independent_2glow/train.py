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
from model import Glow

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description="Glow trainer")
parser.add_argument("--batch", default=16, type=int, help="batch size")
parser.add_argument("--iter", default=200000, type=int, help="maximum iterations")
parser.add_argument("--total_epoch", default=100, type=int, help="maximum epochs")
parser.add_argument(
    "--n_flow", default=32, type=int, help="number of flows in each block"
)
parser.add_argument("--n_block", default=4, type=int, help="number of blocks")
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
parser.add_argument("--img_size", default=128, type=int, help="image size")
parser.add_argument("--temp", default=0.7, type=float, help="temperature of sampling")
parser.add_argument("--n_sample", default=20, type=int, help="number of samples")
# parser.add_argument("path", metavar="PATH", type=str, help="Path to image directory")


def sample_data(path, batch_size, image_size):
    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    dataset = datasets.ImageFolder(path, transform=transform)
    # dataset = datasets.CelebA(root='./dataset', split='train', transform=transform, download=True)
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=4)
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
    # dataset = iter(sample_data(args.path, args.batch, args.img_size))
    dataloader = DataLoader(
        ImageDataset('./dataset/img_align_celeba', (args.img_size, args.img_size)),
        batch_size = args.batch,
        shuffle = True,
        num_workers = 8
    )
    n_bins = 2.0 ** args.n_bits

    z_sample = []
    z_shapes = calc_z_shapes(3, args.img_size, args.n_flow, args.n_block)
    for z in z_shapes:
        z_new = torch.randn(args.n_sample, *z) * args.temp
        z_sample.append(z_new.to(device))

    # with tqdm(range(args.iter)) as pbar:
    for epoch in range(args.total_epoch):
        # for i in pbar:
        for i, imgs in enumerate(dataloader):
            # image, _ = next(dataset)
            # image = image.to(device)
            # image = image * 255

            img_lr, img_hr = imgs["img_lr"].to(device), imgs["img_hr"].to(device)
            img_lr, img_hr = img_lr * 255, img_hr * 255

            if args.n_bits < 8:
                img_lr = torch.floor(img_lr / 2 ** (8 - args.n_bits))

            img_lr = img_lr / n_bins - 0.5

            if i == 0:
                with torch.no_grad():
                    log_p, logdet, _ = model_left.module(
                        img_lr + torch.rand_like(img_lr) / n_bins
                    )

                    continue

            else:
                log_p, logdet, _ = model_left(img_lr + torch.rand_like(img_lr) / n_bins)

            logdet = logdet.mean()

            loss, log_p, log_det = calc_loss(log_p, logdet, args.img_size, n_bins)
            model_left.zero_grad()
            loss.backward()
            # warmup_lr = args.lr * min(1, i * batch_size / (50000 * 10))
            warmup_lr = args.lr
            optimizer_left.param_groups[0]["lr"] = warmup_lr
            optimizer_left.step()

            # pbar.set_description(
            #     f"Loss: {loss.item():.5f}; logP: {log_p.item():.5f}; logdet: {log_det.item():.5f}; lr: {warmup_lr:.7f}"
            # )
            print(
                f"[Epoch {epoch}/{args.total_epoch}, Batch {i}/{len(dataloader)}], Loss: {loss.item():.5f}; logP: {log_p.item():.5f}; logdet: {log_det.item():.5f}; lr: {warmup_lr:.7f}"
            )

            if i % 1 == 0:
                with torch.no_grad():
                    utils.save_image(
                        model_single_left.reverse(z_sample).cpu().data,
                        f"sample/{str(epoch + 1).zfill(6)}.png",
                        normalize=True,
                        nrow=10,
                        range=(-0.5, 0.5),
                    )

            if i % 1 == 0:
                torch.save(
                    model_left.state_dict(), f"checkpoint/model_{str(epoch + 1).zfill(6)}.pt"
                )
                torch.save(
                    optimizer_left.state_dict(), f"checkpoint/optim_{str(epoch + 1).zfill(6)}.pt"
                )


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)

    # left side
    model_single_left = Glow(
        3, args.n_flow, args.n_block, affine=args.affine, conv_lu=not args.no_lu
    )
    model_left = nn.DataParallel(model_single_left)
    # model = model_single
    model_left = model_left.to(device)

    optimizer_left = optim.Adam(model_left.parameters(), lr=args.lr)
 
    # right side
    model_single_right = Glow(
        3, args.n_flow, args.n_block, affine=args.affine, conv_lu=not args.no_lu
    )
    model_right = nn.DataParallel(model_single_right)
    # model = model_single
    model_right = model_right.to(device)

    optimizer_right = optim.Adam(model_right.parameters(), lr=args.lr)

    train(args, model_left, model_right, optimizer_left, optimizer_right)
