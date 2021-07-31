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
from imresize import imresize
from models import create_model
import torch
from utils.util import opt_get
import numpy as np
import pandas as pd
import os
import cv2
from utils import util



def fiFindByWildcard(wildcard):
    return natsort.natsorted(glob.glob(wildcard, recursive=True))


def load_model(conf_path):
    opt = option.parse(conf_path, is_train=False)
    opt['gpu_ids'] = None
    opt = option.dict_to_nonedict(opt)
    model = create_model(opt)

    model_path = opt_get(opt, ['model_path'], None)
    model_path = '/mnt/HDD3_coursework/srdualglow/SRFlow/experiments/train/models/163001_G.pth'
    model.load_sj(load_path=model_path) # network=model.netG)
    # model = model.to('cuda:2')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        network = torch.nn.DataParallel(model)
    network.to(device)

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


def main():
    import torchvision
    from torchvision import transforms

    # conf_path = sys.argv[1]
    # conf = conf_path.split('/')[-1].replace('.yml', '')
    # model, opt = load_model(conf_path)
    from models.SRFlow_model import SRFlowModel
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, help='Path to option YMAL file.')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none',
                        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=True)
    opt = option.dict_to_nonedict(opt)

    # conf_path = sys.argv[1]
    conf_path = 'SRFlow/code/confs/SRFlow_CelebA_4X_seungjae_load_for_test.yml'
    conf = conf_path.split('/')[-1].replace('.yml', '')

    model = SRFlowModel(opt=opt, step=0)

    def sample_data(path, batch_size, image_size):
        transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                # transforms.CenterCrop(image_size),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )

        dataset = torchvision.datasets.ImageFolder(path, transform=transform)
        # print(dataset)
        # dataset = datasets.CelebA(root='./dataset', split='train', transform=transform, download=True)
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

    dataset_lr = iter(sample_data('/mnt/HDD3_coursework/srdualglow/celeba_small_test', 1, 128 // 4))
    dataset_hr = iter(sample_data('/mnt/HDD3_coursework/srdualglow/celeba_small_test', 1, 128))

    dataset = torchvision.datasets.ImageFolder('/mnt/HDD3_coursework/srdualglow/celeba_small_test', transform=None)
    leng = len(dataset)

    this_dir = os.path.dirname(os.path.realpath(__file__))
    test_dir = os.path.join(this_dir, 'results', conf)
    os.makedirs(test_dir, exist_ok=True)
    print(f"Out dir: {test_dir}")

    measure = Measure(use_gpu=False)

    fname = f'measure_full.csv'
    fname_tmp = fname + "_"
    path_out_measures = os.path.join(test_dir, fname_tmp)
    path_out_measures_final = os.path.join(test_dir, fname)

    if os.path.isfile(path_out_measures_final):
        df = pd.read_csv(path_out_measures_final)
    elif os.path.isfile(path_out_measures):
        df = pd.read_csv(path_out_measures)
    else:
        df = None

    scale = opt['scale']

    pad_factor = 2
    with torch.no_grad():
        for idx_test in range(leng):
            lr, _ = next(dataset_lr)
            print(lr.size())
            # lr = lr.cpu()
            hr, _ = next(dataset_hr)
            print(hr.size())
            # hr = hr.cpu()
            # print(lr.size(), hr.size())
            
            # _, _, h, w = hr.size()
            
            # to_pil = transforms.ToPILImage()
            # resize = transforms.Resize((128, 128))
            # resize_32 = transforms.Resize((32, 32))
            # to_ten = transforms.ToTensor()

            # lr = to_ten(resize_32(to_pil(lr[0]))).unsqueeze(0)
            # hr = to_ten(resize(to_pil(hr[0]))).unsqueeze(0)

            # print(lr.size())

            
            heat = opt['heat']  # 0.9 default

            if df is not None and len(df[(df['heat'] == heat) & (df['name'] == idx_test)]) == 1:
                continue

            model.feed_data({'LQ' : lr, 'GT' : hr})
            model.test()

            visuals = model.get_current_visuals()
            '''
            for heat in model.heats:
                for i in range(model.n_sample):
                    sr_img = util.tensor2img(visuals['SR', heat, i])  # uint8                                   int(heat * 100), i))
                    util.save_img(sr_img, f"{test_dir}/sr_{str(idx_test).zfill(6)}_{int(heat * 100)}_{i}.png")
            '''
            # resize_rev = transforms.Resize((h, w))
            visuals_tmp = visuals['SR', heat, 0]
            # visuals_tmp = to_ten(resize_rev(to_pil(visuals['SR', heat, 0]))).unsqueeze(0)
            
            sr_img = util.tensor2img(visuals_tmp)
            
            # sr_img = to_ten(resize_rev(to_pil(sr_img[0]))).unsqueeze(0)



            # sr_t = model.get_sr(lq=lr, heat=heat)

            path_out_sr = os.path.join(test_dir, "{:0.2f}_{:06d}.png".format(heat, idx_test))
            # "{:0.2f}".format(heat).replace('.', ''), 
            # print(path_out_sr)
            if idx_test % (leng // 50) == 0:
                print(f'{idx_test} / {leng}')
            util.save_img(sr_img, path_out_sr)
            
            hr_img = util.tensor2img(hr)

            meas = OrderedDict(conf=conf, heat=heat, name=idx_test)
            meas['PSNR'], meas['SSIM'], meas['LPIPS'] = measure.measure(sr_img, hr_img)

            # lr_reconstruct_rgb = imresize(sr, 1 / opt['scale'])
            # meas['LRC PSNR'] = psnr(lr, lr_reconstruct_rgb)

            str_out = format_measurements(meas)
            print(str_out)
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
