# Modified by Seungjae @ 2021. 07. 31

# Copyright (c) 2020 Huawei Technologies Co., Ltd.
# Licensed under CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike 4.0 International) (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#
# The code is released for academic research use only. For commercial use, please contact Huawei Technologies Co., Ltd.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file contains content licensed by https://github.com/xinntao/BasicSR/blob/master/LICENSE/LICENSE


import glob
import sys
from collections import OrderedDict

from natsort import natsort

import options.options as option
from Measure import Measure, psnr
from models.SRFlow_model import SRFlowModel
from imresize import imresize
from models import create_model
import torch
from torch import nn
from utils.util import opt_get
import numpy as np
import pandas as pd
import os
import cv2
import argparse
from utils import util
from model import Glow, Cond_Glow_Skip

import torchvision
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def fiFindByWildcard(wildcard):
    return natsort.natsorted(glob.glob(wildcard, recursive=True))


def load_model(conf_path):
    opt = option.parse(conf_path, is_train=False)
    opt['gpu_ids'] = None
    opt = option.dict_to_nonedict(opt)
    model = create_model(opt)

    model_path = opt_get(opt, ['model_path'], None)
    model.load_network(load_path=model_path, network=model.netG)
    return model, opt


def predict(model, lr):
    model.feed_data({"LQ": t(lr)}, need_GT=False)
    model.test()
    visuals = model.get_current_visuals(need_GT=False)
    return visuals.get('rlt', visuals.get("SR"))


def t(array): return torch.Tensor(np.expand_dims(array.transpose([2, 0, 1]), axis=0).astype(np.float32)) / 255


def rgb(t): return (
        np.clip((t[0] if len(t.shape) == 4 else t).detach().cpu().numpy().transpose([1, 2, 0]), 0, 1) * 255).astype(
    np.uint8)


def imread(path):
    return cv2.imread(path)[:, :, [2, 1, 0]]


def imwrite(path, img):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, img[:, :, [2, 1, 0]])


def imCropCenter(img, size):
    h, w, c = img.shape

    h_start = max(h // 2 - size // 2, 0)
    h_end = min(h_start + size, h)

    w_start = max(w // 2 - size // 2, 0)
    w_end = min(w_start + size, w)

    return img[h_start:h_end, w_start:w_end]


def impad(img, top=0, bottom=0, left=0, right=0, color=255):
    return np.pad(img, [(top, bottom), (left, right), (0, 0)], 'reflect')


def sample_data(path, batch_size, image_size):
    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ]
    )

    dataset = torchvision.datasets.ImageFolder(path, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, shuffle=False, batch_size=batch_size)
    loader = iter(loader)

    while True:
        try:
            yield next(loader)

        except StopIteration:
            loader = torch.utils.data.DataLoader(
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


def calc_inp_shapes(n_channels, image_size, n_blocks):
    # calculates z shapes (inputs) after SQUEEZE operation (before Block operations) - e.g. channels: 12, 24, 48, 96
    def calc_z_shapes(n_channel, image_size, n_block):
        # calculates shapes of z's after SPLIT operation (after Block operations) - e.g. channels: 6, 12, 24, 96
        z_shapes = []
        for i in range(n_block - 1):
            image_size = (image_size[0] // 2, image_size[1] // 2)
            n_channel = n_channel * 2

            shape = (n_channel, *image_size)
            z_shapes.append(shape)

        # for the very last block where we have no split operation
        image_size = (image_size[0] // 2, image_size[1] // 2)
        shape = (n_channel * 4, *image_size)
        z_shapes.append(shape)
        return z_shapes

    z_shapes = calc_z_shapes(n_channels, image_size, n_blocks)
    input_shapes = []
    for i in range(len(z_shapes)):
        if i < len(z_shapes) - 1:
            channels = z_shapes[i][0] * 2
            input_shapes.append((channels, z_shapes[i][1], z_shapes[i][2]))
        else:
            input_shapes.append((z_shapes[i][0], z_shapes[i][1], z_shapes[i][2]))
    return input_shapes


def calc_cond_shapes(n_channels, image_size, n_blocks):
    # computes additional channels dimensions based on additional conditions: left input + condition
    input_shapes = calc_inp_shapes(n_channels, image_size, n_blocks)
    cond_shapes = []
    for block_idx in range(len(input_shapes)):
        shape = [input_shapes[block_idx][0], input_shapes[block_idx][1], input_shapes[block_idx][2]]  # from left glow
        cond_shapes.append(tuple(shape))
    return cond_shapes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default='./confs/SRFlow_CelebA_4X_seungjae_load_for_test.yml',
                        help='Path to option YMAL file.')
    parser.add_argument('--p', type=str, default='./celeba_small_test',
                        help='Path to celeba_small_test')
    parser.add_argument('--exp_name', type=str,
                        default='SRDualGlow',
                        help='exp name')

    args = parser.parse_args()

    ### For SRFlow
    opt = option.parse(args.opt, is_train=True)
    opt = option.dict_to_nonedict(opt)

    conf_path = 'SRFlow_seungjae/code/confs/SRFlow_CelebA_4X_seungjae_load_for_test.yml'
    conf = conf_path.split('/')[-1].replace('.yml', '')

    n_bits = 5
    n_bins = 2.0 ** n_bits
    temp = 0.8
    img_size = 128
    n_block, n_flow = 4, 32
    no_lu = False
    affine = True
    tr_resize = transforms.Resize((img_size, img_size))
    left_glow_params = './srdualglow_models/model_lr_040001.pt'
    right_glow_params = './srdualglow_models/model_hr_190001.pt'
    input_shapes = calc_inp_shapes(3, [img_size, img_size], n_block)
    cond_shapes = calc_cond_shapes(3, [img_size, img_size], n_block)

    model_single_left = Glow(
        3, n_flow, n_block, affine=False, conv_lu=not no_lu
    )
    model_left = nn.DataParallel(model_single_left)
    model_left = model_left.to(device)
    model_left.load_state_dict(torch.load(left_glow_params))
    model_left.eval()

    model_single_right = Cond_Glow_Skip(
        3, n_flow, n_block, input_shapes, cond_shapes, affine=affine, conv_lu=not no_lu
    )
    model_right = nn.DataParallel(model_single_right)
    model_right = model_right.to(device)
    model_right.load_state_dict(torch.load(right_glow_params))
    model_right.eval()

    ### Load dataset

    dataset_lr = iter(sample_data(args.p, 1, 128 // 4))
    dataset_hr = iter(sample_data(args.p, 1, 128))

    dataset = torchvision.datasets.ImageFolder(args.p)
    leng = len(dataset)

    test_dir = f'./{args.exp_name}_results'
    os.makedirs(test_dir, exist_ok=True)
    print(f"Out dir: {test_dir}")

    measure = Measure(use_gpu=False)

    fname = f'measure_full.csv'
    fname_tmp = fname + "_"
    path_out_measures = os.path.join(test_dir, fname_tmp)
    path_out_measures_final = os.path.join(test_dir, fname)
    print(path_out_measures)

    if os.path.isfile(path_out_measures_final):
        df = pd.read_csv(path_out_measures_final)
    elif os.path.isfile(path_out_measures):
        df = pd.read_csv(path_out_measures)
    else:
        df = None

    for idx_test in range(leng):
        image_lr, _ = next(dataset_lr)
        image_lr = image_lr.to(device)
        image_hr, _ = next(dataset_hr)
        image_hr = image_hr.to(device)

        z_sample_hr = []
        z_shapes_hr = calc_z_shapes(3, img_size, n_flow, n_block)
        for z in z_shapes_hr:
            z_new = torch.randn(len(image_lr), *z) * temp
            z_sample_hr.append(z_new.to(device))

        image_lr = image_lr * 255
        if n_bits < 8:
            image_lr = torch.floor(image_lr / 2 ** (8 - n_bits))
        image_lr = image_lr / n_bins - 0.5
        image_hr = image_hr * 255
        if n_bits < 8:
            image_hr = torch.floor(image_hr / 2 ** (8 - n_bits))
        image_hr = image_hr / n_bins - 0.5
        image_lr_resize = tr_resize(image_lr)

        with torch.no_grad():
            left_glow_out = model_left.module(image_lr + torch.rand_like(image_lr) / n_bins)
            image_sr = model_single_right.reverse(z_sample_hr, reconstruct=False,
                                                  left_glow_out=left_glow_out,
                                                  image_lr_resize=image_lr_resize)

        ### Inference part (Currently for SRDualGlow)
        print(image_hr.size(), torch.max(image_hr), torch.min(image_hr))
        hr = rgb(image_hr + 0.5)  # To make numpy array
        sr = rgb(image_sr + 0.5)  # To make numpy array

        # IMSAVE
        path_out_sr = f'{test_dir}/{idx_test:06d}.png'
        imwrite(path_out_sr, sr)

        # MEASURE
        meas = OrderedDict(conf=conf, heat=temp, name=idx_test)
        meas['PSNR'], meas['SSIM'], meas['LPIPS'] = measure.measure(sr, hr)

        str_out = format_measurements(meas)
        print(str_out)

        # SAVE CSV
        df = pd.DataFrame([meas]) if df is None else pd.concat([pd.DataFrame([meas]), df])

        df.to_csv(path_out_measures + "_", index=False)
        os.rename(path_out_measures + "_", path_out_measures)

    df.to_csv(path_out_measures, index=False)
    os.rename(path_out_measures, path_out_measures_final)

    str_out = format_measurements(df.mean())
    print(f"Results in: {path_out_measures_final}")
    print('Mean: ' + str_out)


def format_measurements(meas):
    s_out = []
    for k, v in meas.items():
        v = f"{v:0.2f}" if isinstance(v, float) else v
        s_out.append(f"{k}: {v}")
    str_out = ", ".join(s_out)
    return str_out


if __name__ == "__main__":
    main()
