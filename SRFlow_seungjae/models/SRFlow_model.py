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

import logging
from collections import OrderedDict
from typing_extensions import runtime
from utils.util import get_resume_paths, opt_get

import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import models.networks as networks
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
from torch.autograd import Variable
import numpy as np
import copy
from torchvision.utils import save_image
from models.modules.RRDBNet_arch import Discriminator

logger = logging.getLogger('base')


class SRFlowModel(BaseModel):
    def __init__(self, opt, step):
        super(SRFlowModel, self).__init__(opt)
        self.opt = opt

        self.heats = opt['val']['heats']
        self.n_sample = opt['val']['n_sample']
        self.hr_size = opt_get(opt, ['datasets', 'train', 'center_crop_hr_size'])
        self.hr_size = 160 if self.hr_size is None else self.hr_size
        self.lr_size = self.hr_size // opt['scale']

        # loss
        self.loss_pixel = 0.0
        self.loss_GAN = 0.0
        self.loss_G = 0.0
        self.loss_D = 0.0

        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt['train']

        # define network and load pretrained models
        self.netG = networks.define_Flow(opt, step).to(self.device)
        if opt['dist']:
            self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()])
        else:
            self.netG = DataParallel(self.netG)
        
        lr_size = opt['hr_size'] // opt['scale']
        hr_size = opt['hr_size']
        self.netG.module.Discriminator = Discriminator((320, lr_size,lr_size)).cuda() # this part should be modified
        self.netG.module.Discriminator_rgb = Discriminator((opt['network_G']['in_nc'],hr_size,hr_size)).cuda() # this part should be modified
        # print network
        self.print_network()
        '''
        if opt_get(opt, ['path', 'resume_state'], 1) is not None:
            self.load()
        else:
            print("WARNING: skipping initial loading, due to resume_state None")
        '''

        if self.is_train:
            self.netG.train()

            self.init_optimizer_and_scheduler(train_opt)
            self.log_dict = OrderedDict()

    def to(self, device):
        self.device = device
        self.netG.to(device)

    def init_optimizer_and_scheduler(self, train_opt):
        # optimizers
        self.optimizers = []
        wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
        optim_params_RRDB = []
        optim_params_RRDB_O = []
        optim_params_other = []
        for k, v in self.netG.named_parameters():  # can optimize for a part of the model
            print(k, v.requires_grad)
            if v.requires_grad:
                if '.RRDB.' in k:
                    optim_params_RRDB.append(v)
                    print('opt', k)
                elif '.RRDB_O.' in k:
                    optim_params_RRDB_O.append(v)
                    print('opt_rrdbo', k)
                else:
                    optim_params_other.append(v)
                if self.rank <= 0:
                    logger.warning('Params [{:s}] will not optimize.'.format(k))

        print('rrdb params', len(optim_params_RRDB))

        self.optimizer_G = torch.optim.Adam(
            [
                {"params": optim_params_other, "lr": train_opt['lr_G'], 'beta1': train_opt['beta1'],
                 'beta2': train_opt['beta2'], 'weight_decay': wd_G},
                # {"params": optim_params_RRDB, "lr": train_opt.get('lr_RRDB', train_opt['lr_G']),
                #  'beta1': train_opt['beta1'],
                #  'beta2': train_opt['beta2'], 'weight_decay': wd_G}
            ],
        )

        self.optimizer_G_RRDB_O = torch.optim.Adam(
            [
                {"params": optim_params_other, "lr": train_opt['lr_G'], 'beta1': train_opt['beta1'],
                 'beta2': train_opt['beta2'], 'weight_decay': wd_G},
                {"params": optim_params_RRDB_O, "lr": train_opt.get('lr_RRDB', train_opt['lr_G']),
                 'beta1': train_opt['beta1'],
                 'beta2': train_opt['beta2'], 'weight_decay': wd_G}
            ],
        )

        # self.optimizer_PG = torch.optim.Adam(self.netG.module.RRDB_O.parameters(), lr=2e-4, betas=(0.9, 0.999))
        self.optimizer_PD = torch.optim.Adam(self.netG.module.Discriminator.parameters(), lr=2e-4, betas=(0.9, 0.999))
        self.optimizer_PD_rgb = torch.optim.Adam(self.netG.module.Discriminator_rgb.parameters(), lr=2e-4, betas=(0.9, 0.999))

        self.optimizers.append(self.optimizer_G)
        # schedulers
        if train_opt['lr_scheme'] == 'MultiStepLR':
            for optimizer in self.optimizers:
                self.schedulers.append(
                    lr_scheduler.MultiStepLR_Restart(optimizer, train_opt['lr_steps'],
                                                     # restarts=train_opt['restarts'],
                                                     # weights=train_opt['restart_weights'],
                                                     gamma=train_opt['lr_gamma'],
                                                     # clear_state=train_opt['clear_state'],
                                                     lr_steps_invese=train_opt.get('lr_steps_inverse', [])))
        elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
            for optimizer in self.optimizers:
                self.schedulers.append(
                    lr_scheduler.CosineAnnealingLR_Restart(
                        optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'],
                        restarts=train_opt['restarts'], weights=train_opt['restart_weights']))
        else:
            raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

    def add_optimizer_and_scheduler_RRDB(self, train_opt):
        # optimizers
        assert len(self.optimizers) == 1, self.optimizers
        assert len(self.optimizer_G.param_groups[1]['params']) == 0, self.optimizer_G.param_groups[1]
        for k, v in self.netG.named_parameters():  # can optimize for a part of the model
            if v.requires_grad:
                if '.RRDB.' in k:
                    self.optimizer_G.param_groups[1]['params'].append(v)
        assert len(self.optimizer_G.param_groups[1]['params']) > 0

    def feed_data(self, data, need_GT=True):
        self.var_L = data['clean_lr'].to(self.device)  # Pseudo LR
        self.real_L = data['real_lr'].to(self.device)  # Real LR
        if need_GT:
            self.real_H = data['real_hr'].to(self.device)  # Real HR

    def save_encoder(self, epoch, exp_name):
        torch.save(self.netG.module.RRDB_O.state_dict(), f'./saved_models/{exp_name}/encoder{epoch}.pth')

    def copy_rrdb(self):
        self.netG.module.RRDB_O = copy.deepcopy(self.netG.module.RRDB)
        freezed_layers = ['HRconv', 'conv_last']

        rrdb_o_param_list = []
        
        # for p in self.netG.module.RRDB_O.parameters():
        #     p.requires_grad = True

        for n, p in self.netG.module.RRDB_O.named_parameters():
            p.requires_grad = True
            if sum([f in n for f in freezed_layers]) == 0:
                rrdb_o_param_list.append(p)

            # if 
            # rrdb_o_param_list.append(p)

        
        # for p in self.netG.module.RRDB.parameters():
                # p.requires_grad = False
        self.optimizer_PG = torch.optim.Adam(rrdb_o_param_list, lr=2e-4, betas=(0.9, 0.999))
        # self.optimizer_PG = torch.optim.Adam(self.netG.module.RRDB_O.parameters(), lr=2e-4, betas=(0.9, 0.999))

    def unified_forward(self, step, Tensor):

        criterion_GAN = torch.nn.BCEWithLogitsLoss().to(self.device)
        criterion_pixel = torch.nn.L1Loss().to(self.device)
        

        
        train_RRDB_delay = opt_get(self.opt, ['network_G', 'train_RRDB_delay'])
        if train_RRDB_delay is not None and step > int(train_RRDB_delay * self.opt['train']['niter']) \
                and not self.netG.module.RRDB_training:
            if self.netG.module.set_rrdb_training(True):
                self.add_optimizer_and_scheduler_RRDB(self.opt['train'])
        # self.print_rrdb_state()

        self.netG.train()
        self.log_dict = OrderedDict()
        self.optimizer_G.zero_grad()

        '''
        losses = {}
        weight_fl = opt_get(self.opt, ['train', 'weight_fl'])
        weight_fl = 1 if weight_fl is None else weight_fl
        if weight_fl > 0:
            z, nll, y_logits = self.netG(gt=self.real_H, lr=self.var_L, reverse=False)
            nll_loss = torch.mean(nll)
            losses['nll_loss'] = nll_loss * weight_fl

        weight_l1 = opt_get(self.opt, ['train', 'weight_l1']) or 0
        if weight_l1 > 0:
            z = self.get_z(heat=0, seed=None, batch_size=self.var_L.shape[0], lr_shape=self.var_L.shape)
            sr, logdet = self.netG(lr=self.var_L, z=z, eps_std=0, reverse=True, reverse_with_grad=True)
            l1_loss = (sr - self.real_H).abs().mean()
            losses['l1_loss'] = l1_loss * weight_l1

        '''
        z = self.get_z(heat=0, seed=None, batch_size=self.var_L.shape[0], lr_shape=self.var_L.shape)
        clean_hr, _, target =self.netG(lr=self.var_L, z=z, eps_std=0, reverse=True, reverse_with_grad=True, clean = True, flag=True)
        real_hr, _, output =self.netG(lr=self.real_L, z=z, eps_std=0, reverse=True, reverse_with_grad=True, clean = False, flag=True)
        '''
        nll_loss = sum(losses.values())
        nll_loss.backward()
        self.optimizer_G.step()
        '''


        # ---------------------
        # Traing image loss of Generator
        # ---------------------

        layers = ['fea_up1', 'fea_up2', 'fea_up4']
        layer_idx = 0
        cur_layer = layers[layer_idx]

        '''
        layers = ['fea_up1', 'fea_up2', 'fea_up4'] # each size is (320, , )

        # print(output['fea_up1'].shape)
        # print(output['fea_up2'].shape)
        # print(output['fea_up4'].shape)

        # exit()
        layer_idx = np.random.randint(3)
        cur_layer = layers[layer_idx]
        '''
    

        spatial_size = []
        for i, size in enumerate(self.netG.module.Discriminator.output_shape):
            if i==0:
                spatial_size.append(size)    
            else:
                spatial_size.append(size * 2**(layer_idx))

        target = target[cur_layer]
        output = output[cur_layer]


        valid = Variable(Tensor(np.ones((output.size(0), *spatial_size))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((output.size(0), *spatial_size))), requires_grad=False)

        valid_rgb = Variable(Tensor(np.ones((output.size(0), *self.netG.module.Discriminator_rgb.output_shape))), requires_grad=False)
        fake_rgb = Variable(Tensor(np.zeros((output.size(0), *self.netG.module.Discriminator_rgb.output_shape))), requires_grad=False)

        idt_output = self.netG.module.rrdbPreprocessing(self.var_L, clean=False)[cur_layer] 


        self.loss_pixel = criterion_pixel(idt_output, target) * 0

        pred_real = self.netG.module.Discriminator(target).detach()
        pred_fake = self.netG.module.Discriminator(output)


        self.loss_GAN = criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), valid)

        
        self.optimizer_G_RRDB_O.zero_grad()
        self.loss_GAN.backward()
        self.optimizer_G_RRDB_O.step()


        # -----------------------------
        # RGB Discriminator
        # -----------------------------

        clean_hr, _, target =self.netG(lr=self.var_L, z=z, eps_std=0, reverse=True, reverse_with_grad=True, clean = True, flag=True)
        real_hr, _, output =self.netG(lr=self.real_L, z=z, eps_std=0, reverse=True, reverse_with_grad=True, clean = False, flag=True)

        target = target[cur_layer]
        output = output[cur_layer]

        pred_real_rgb = self.netG.module.Discriminator_rgb(real_hr).detach()
        pred_fake_rgb = self.netG.module.Discriminator_rgb(clean_hr)
        # print(real_hr.size(), clean_hr.size()) [4, 3, 128, 128]

        self.loss_GAN_rgb = criterion_GAN(pred_fake_rgb - pred_real_rgb.mean(0, keepdim=True), valid_rgb)

        self.optimizer_G_RRDB_O.zero_grad()
        self.loss_GAN_rgb.backward()
        self.optimizer_G_RRDB_O.step()
        

        # ---------------------
        #  Train Discriminator
        # ---------------------

        self.optimizer_PD.zero_grad()
        pred_real = self.netG.module.Discriminator(target)
        pred_fake = self.netG.module.Discriminator(output.detach())

        # Adversarial loss for real and fake images (relativistic average GAN)
        loss_real = criterion_GAN(pred_real - pred_fake.mean(0, keepdim=True), valid)
        loss_fake = criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), fake)

        # Total loss
        self.loss_D = (loss_real + loss_fake) / 2

        self.loss_D.backward()
        self.optimizer_PD.step()


        clean_hr, _, target =self.netG(lr=self.var_L, z=z, eps_std=0, reverse=True, reverse_with_grad=True, clean = True, flag=True)
        real_hr, _, output =self.netG(lr=self.real_L, z=z, eps_std=0, reverse=True, reverse_with_grad=True, clean = False, flag=True)

        self.optimizer_PD_rgb.zero_grad()

        pred_real_rgb = self.netG.module.Discriminator_rgb(real_hr)
        pred_fake_rgb = self.netG.module.Discriminator_rgb(clean_hr.detach())

        loss_real_rgb = criterion_GAN(pred_real_rgb - pred_fake_rgb.mean(0, keepdim=True), valid_rgb)
        loss_fake_rgb = criterion_GAN(pred_fake_rgb - pred_real_rgb.mean(0, keepdim=True), fake_rgb)

        self.loss_D_rgb = (loss_real_rgb + loss_fake_rgb) / 2

        self.loss_D_rgb.backward()
        self.optimizer_PD_rgb.step()

        # return nll_loss, self.loss_pixel, self.loss_GAN, self.loss_GAN_rgb, self.loss_D, self.loss_D_rgb
        return torch.zeros(1), self.loss_pixel, self.loss_GAN, self.loss_D

    def pre_forward(self, Tensor):
        num_levels = 3
        self.netG.train()

        layers = ['fea_up0', 'fea_up1', 'fea_up2', 'fea_up4']
        layer_idx = 1
        cur_layer = layers[layer_idx]

        # Configure model input
        self.var_L = Variable(self.var_L.type(Tensor))
        self.real_L = Variable(self.real_L.type(Tensor))


        # Set model input & tnetGarget
        target = self.netG.module.RRDB(self.var_L, get_steps=True)[cur_layer].detach()

        # rgb_target = self.netG.module.RRDB(self.var_L, get_steps=True)['out'].detach()

        # rgb_target = self.netG.module.RRDB(self.var_L).detach()

        criterion_GAN = torch.nn.BCEWithLogitsLoss().to(self.device)
        criterion_pixel = torch.nn.L1Loss().to(self.device)

        # ------------------
        #  Train Generators
        # ------------------
        self.optimizer_PG.zero_grad()

        output = self.netG.module.RRDB_O(self.real_L, get_steps=True)[cur_layer]
        idt_output = self.netG.module.RRDB_O(self.var_L, get_steps=True)[cur_layer]

        # rgb_output = self.netG.module.RRDB_O(self.real_L, get_steps=True)['out']

        z = self.get_z(heat=0, seed=None, batch_size=self.var_L.shape[0], lr_shape=self.var_L.shape)

        rgb_target, _ = self.netG(lr=self.var_L, z=z, eps_std=0, reverse=True, clean=True)
        rgb_output, _ = self.netG(lr=self.real_L, z=z, eps_std=0, reverse=True, clean=False)

        # print(rgb_target.min())
        # print(rgb_target.max())
        # print('---- target -------')
        # print(rgb_output.min())
        # print(rgb_output.max())

        for i, p in enumerate(self.netG.module.flowUpsamplerNet.parameters()):
            if i == 0:
                print(p.flatten()[:3])

        rgb_output = rgb_output.detach()


        # # Debugging the loading pretrained model
        # save_image(self.var_L, 'input_sample.png')
        # save_image(rgb_output, 'output_sample.png')
        # save_image(rgb_target, 'target_sample.png')
        # exit()

        # Adversarial ground truths
        
        spatial_size = []
        for i, size in enumerate(self.netG.module.Discriminator.output_shape):
            if i==0:
                spatial_size.append(size)    
            else:
                spatial_size.append(size * 2**(layer_idx))

        valid = Variable(Tensor(np.ones((output.size(0), *spatial_size))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((output.size(0), *spatial_size))), requires_grad=False)

        valid_rgb = Variable(Tensor(np.ones((output.size(0), *self.netG.module.Discriminator_rgb.output_shape))), requires_grad=False)
        fake_rgb = Variable(Tensor(np.zeros((output.size(0), *self.netG.module.Discriminator_rgb.output_shape))), requires_grad=False)

        # Total generator loss
        self.loss_pixel = criterion_pixel(idt_output, target)

        pred_real = self.netG.module.Discriminator(target).detach()
        pred_fake = self.netG.module.Discriminator(output)

        pred_real_rgb = self.netG.module.Discriminator_rgb(rgb_target).detach()
        pred_fake_rgb = self.netG.module.Discriminator_rgb(rgb_output)

        self.loss_GAN = criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), valid)
        self.loss_GAN_rgb = criterion_GAN(pred_fake_rgb - pred_real_rgb.mean(0, keepdim=True), valid_rgb)

        self.loss_G =  0.01 * (self.loss_GAN + self.loss_GAN_rgb) + self.loss_pixel

        # if self.loss_GAN > self.loss_D + 0.4:
        self.loss_G.backward()
        self.optimizer_PG.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        self.optimizer_PD.zero_grad()

        pred_real = self.netG.module.Discriminator(target)
        pred_fake = self.netG.module.Discriminator(output.detach())

        # Adversarial loss for real and fake images (relativistic average GAN)
        loss_real = criterion_GAN(pred_real - pred_fake.mean(0, keepdim=True), valid)
        loss_fake = criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), fake)

        # Total loss
        self.loss_D = (loss_real + loss_fake) / 2

        self.loss_D.backward()
        self.optimizer_PD.step()



        self.optimizer_PD_rgb.zero_grad()

        pred_real_rgb = self.netG.module.Discriminator_rgb(rgb_target)
        pred_fake_rgb = self.netG.module.Discriminator_rgb(rgb_output.detach())

        loss_real_rgb = criterion_GAN(pred_real_rgb - pred_fake_rgb.mean(0, keepdim=True), valid_rgb)
        loss_fake_rgb = criterion_GAN(pred_fake_rgb - pred_real_rgb.mean(0, keepdim=True), fake_rgb)

        self.loss_D_rgb = (loss_real_rgb + loss_fake_rgb) / 2

        self.loss_D_rgb.backward()
        self.optimizer_PD_rgb.step()

        return self.loss_GAN, self.loss_GAN_rgb, self.loss_pixel, self.loss_D, self.loss_D_rgb

    def optimize_parameters(self, step):
        train_RRDB_delay = opt_get(self.opt, ['network_G', 'train_RRDB_delay'])
        if train_RRDB_delay is not None and step > int(train_RRDB_delay * self.opt['train']['niter']) \
                and not self.netG.module.RRDB_training:
            if self.netG.module.set_rrdb_training(True):
                self.add_optimizer_and_scheduler_RRDB(self.opt['train'])
        # self.print_rrdb_state()

        self.netG.train()
        self.log_dict = OrderedDict()
        self.optimizer_G.zero_grad()

        losses = {}
        weight_fl = opt_get(self.opt, ['train', 'weight_fl'])
        weight_fl = 1 if weight_fl is None else weight_fl
        if weight_fl > 0:
            z, nll, y_logits = self.netG(gt=self.real_H, lr=self.var_L, reverse=False)
            nll_loss = torch.mean(nll)
            losses['nll_loss'] = nll_loss * weight_fl

        weight_l1 = opt_get(self.opt, ['train', 'weight_l1']) or 0
        if weight_l1 > 0:
            z = self.get_z(heat=0, seed=None, batch_size=self.var_L.shape[0], lr_shape=self.var_L.shape)
            sr, logdet = self.netG(lr=self.var_L, z=z, eps_std=0, reverse=True, reverse_with_grad=True)
            l1_loss = (sr - self.real_H).abs().mean()
            losses['l1_loss'] = l1_loss * weight_l1

        total_loss = sum(losses.values())
        total_loss.backward()
        self.optimizer_G.step()

        mean = total_loss.item()
        return mean

    def print_rrdb_state(self):
        for name, param in self.netG.module.named_parameters():
            if "RRDB.conv_first.weight" in name:
                print(name, param.requires_grad, param.data.abs().sum())
        print('params', [len(p['params']) for p in self.optimizer_G.param_groups])

    def test(self):
        self.netG.eval()
        self.fake_H_clean = {}
        self.fake_H_real = {}
        self.fake_H_real_initial = {}
        for heat in self.heats:
            for i in range(self.n_sample):
                z = self.get_z(heat, seed=None, batch_size=self.var_L.shape[0], lr_shape=self.var_L.shape)
                with torch.no_grad():
                    self.fake_H_clean[(heat, i)], _ = self.netG(lr=self.var_L, z=z, eps_std=heat, reverse=True, clean=True)
                    self.fake_H_real[(heat, i)], _ = self.netG(lr=self.real_L, z=z, eps_std=heat, reverse=True, clean=False)
                    self.fake_H_real_initial[(heat, i)], _ = self.netG(lr=self.real_L, z=z, eps_std=heat, reverse=True, clean=True)
        with torch.no_grad():
            _, nll, _ = self.netG(gt=self.real_H, lr=self.var_L, reverse=False)
        self.netG.train()
        return nll.mean().item()

    def get_encode_nll(self, lq, gt):
        self.netG.eval()
        with torch.no_grad():
            _, nll, _ = self.netG(gt=gt, lr=lq, reverse=False)
        self.netG.train()
        return nll.mean().item()

    def get_sr(self, lq, heat=None, seed=None, z=None, epses=None):
        return self.get_sr_with_z(lq, heat, seed, z, epses)[0]

    def get_encode_z(self, lq, gt, epses=None, add_gt_noise=True):
        self.netG.eval()
        with torch.no_grad():
            z, _, _ = self.netG(gt=gt, lr=lq, reverse=False, epses=epses, add_gt_noise=add_gt_noise)
        self.netG.train()
        return z

    def get_encode_z_and_nll(self, lq, gt, epses=None, add_gt_noise=True):
        self.netG.eval()
        with torch.no_grad():
            z, nll, _ = self.netG(gt=gt, lr=lq, reverse=False, epses=epses, add_gt_noise=add_gt_noise)
        self.netG.train()
        return z, nll

    def get_sr_with_z(self, lq, heat=None, seed=None, z=None, epses=None):
        self.netG.eval()

        z = self.get_z(heat, seed, batch_size=lq.shape[0], lr_shape=lq.shape) if z is None and epses is None else z

        with torch.no_grad():
            sr, logdet = self.netG(lr=lq, z=z, eps_std=heat, reverse=True, epses=epses)
        self.netG.train()
        return sr, z

    def get_z(self, heat, seed=None, batch_size=1, lr_shape=None):
        if seed: torch.manual_seed(seed)
        if opt_get(self.opt, ['network_G', 'flow', 'split', 'enable']):
            C = self.netG.module.flowUpsamplerNet.C
            H = int(self.opt['scale'] * lr_shape[2] // self.netG.module.flowUpsamplerNet.scaleH)
            W = int(self.opt['scale'] * lr_shape[3] // self.netG.module.flowUpsamplerNet.scaleW)
            z = torch.normal(mean=0, std=heat, size=(batch_size, C, H, W)) if heat > 0 else torch.zeros(
                (batch_size, C, H, W))
        else:
            L = opt_get(self.opt, ['network_G', 'flow', 'L']) or 3
            fac = 2 ** (L - 3)
            z_size = int(self.lr_size // (2 ** (L - 3)))
            z = torch.normal(mean=0, std=heat, size=(batch_size, 3 * 8 * 8 * fac * fac, z_size, z_size))
        return z

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict['clean_lr'] = self.var_L.detach()[0].float().cpu()
        out_dict['real_lr'] = self.real_L.detach()[0].float().cpu()
        for heat in self.heats:
            for i in range(self.n_sample):
                out_dict[('SR_clean', heat, i)] = self.fake_H_clean[(heat, i)].detach()[0].float().cpu()
                out_dict[('SR_real', heat, i)] = self.fake_H_real[(heat, i)].detach()[0].float().cpu()
                out_dict[('SR_real_initial', heat, i)] = self.fake_H_real_initial[(heat, i)].detach()[0].float().cpu()
        if need_GT:
            out_dict['real_hr'] = self.real_H.detach()[0].float().cpu()
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel) or isinstance(self.netG, DistributedDataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)

    def load(self):
        _, get_resume_model_path = get_resume_paths(self.opt)
        if get_resume_model_path is not None:
            self.load_network(get_resume_model_path, self.netG, strict=True, submodule=None)
            return

        load_path_G = self.opt['path']['pretrain_model_G']
        load_submodule = self.opt['path']['load_submodule'] if 'load_submodule' in self.opt['path'].keys() else 'RRDB'
        if load_path_G is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, self.opt['path'].get('strict_load', True),
                              submodule=load_submodule)

    def save(self, iter_label, exp_name):
        self.save_network(self.netG, 'G', iter_label, exp_name)
