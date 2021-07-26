import torch
from torch import nn
from torch.nn import functional as F
from math import log, pi, exp
import numpy as np
from scipy import linalg as la
from utils import *
from cond_net import *
import math

logabs = lambda x: torch.log(torch.abs(x))

RESIZE_CHOICE = 1
RESIZE_SCALE = 4

def set_resize(args):
    global RESIZE_CHOICE
    global RESIZE_SCALE
    RESIZE_CHOICE = args.resize_choice
    RESIZE_SCALE = args.resize_scale

class resize(nn.Module):
    def __init__(self, cond_shape):
        super().__init__()
        global RESIZE_CHOICE
        global RESIZE_SCALE

        if RESIZE_CHOICE == 1:
            num_upsample = int(math.log2(RESIZE_SCALE))
            upsample_layers = []
            filters = cond_shape[0]
            for _ in range(num_upsample):
                upsample_layers += [
                    nn.Conv2d(filters, filters * 4, kernel_size=3, stride=1, padding=1),
                    nn.LeakyReLU(),
                    nn.PixelShuffle(upscale_factor=2),
                ]
            upsample_layers += [nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(),
                nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1),
                ]
            upsampling = nn.Sequential(*upsample_layers)
            self.upsampling = upsampling.cuda()


    def forward(self, inp):
        if RESIZE_CHOICE == 0:
            return torch.nn.functional.interpolate(inp, scale_factor=RESIZE_SCALE, mode='bilinear')
        elif RESIZE_CHOICE == 1:
            return self.upsampling(inp)




class ActNorm(nn.Module):
    def __init__(self, in_channel, logdet=True):
        super().__init__()

        self.loc = nn.Parameter(torch.zeros(1, in_channel, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, in_channel, 1, 1))

        self.register_buffer("initialized", torch.tensor(0, dtype=torch.uint8))
        self.logdet = logdet

    def initialize(self, input):
        with torch.no_grad():
            flatten = input.permute(1, 0, 2, 3).contiguous().view(input.shape[1], -1)
            mean = (
                flatten.mean(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )
            std = (
                flatten.std(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )

            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, input):
        _, _, height, width = input.shape

        if self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(1)

        log_abs = logabs(self.scale)

        logdet = height * width * torch.sum(log_abs)

        if self.logdet:
            return self.scale * (input + self.loc), logdet

        else:
            return self.scale * (input + self.loc)

    def reverse(self, output):
        return output / self.scale - self.loc

class Cond_Actnorm(nn.Module):
    def __init__(self, cond_shape, inp_shape):
        super().__init__()
        self.cond_net = ActCondNet(cond_shape, inp_shape)

        print_params = False
        if print_params:
            total_params = sum(p.numel() for p in self.cond_net.parameters())
            print('ActNormConditional CondNet params:', total_params)

        self.resize = resize(cond_shape)

    def forward(self, inp, condition):
        # condition = torch.nn.functional.interpolate(condition, inp.size()[2:], mode='bilinear')
        condition = self.resize(condition)
        cond_out = self.cond_net(condition, inp)  # output shape (B, 2, C)
        s = cond_out[:, 0, :].unsqueeze(2).unsqueeze(3)  # s, t shape (B, C, 1, 1)
        t = cond_out[:, 1, :].unsqueeze(2).unsqueeze(3)

        # computing log determinant
        _, _, height, width = inp.shape  # input of shape [bsize, in_channel, h, w]
        scale_logabs = logabs(s).mean(dim=0, keepdim=True)  # mean over batch - shape: (1, C, 1, 1)
        log_det = height * width * torch.sum(scale_logabs)  # scalar value
        return s * (inp + t), log_det

    def reverse(self, out, condition):
        # condition = torch.nn.functional.interpolate(condition, out.size()[2:], mode='bilinear')
        condition = self.resize(condition)
        cond_out = self.cond_net(condition)  # output shape (B, 2, C)
        s = cond_out[:, 0, :].unsqueeze(2).unsqueeze(3)  # s, t shape (B, C, 1, 1)
        t = cond_out[:, 1, :].unsqueeze(2).unsqueeze(3)
        return (out / s) - t


class InvConv2d(nn.Module):
    def __init__(self, in_channel):
        super().__init__()

        weight = torch.randn(in_channel, in_channel)
        q, _ = torch.qr(weight)
        weight = q.unsqueeze(2).unsqueeze(3)
        self.weight = nn.Parameter(weight)

    def forward(self, input):
        _, _, height, width = input.shape

        out = F.conv2d(input, self.weight)
        logdet = (
            height * width * torch.slogdet(self.weight.squeeze().double())[1].float()
        )

        return out, logdet

    def reverse(self, output):
        return F.conv2d(
            output, self.weight.squeeze().inverse().unsqueeze(2).unsqueeze(3)
        )


class InvConv2dLU(nn.Module):
    def __init__(self, in_channel):
        super().__init__()

        weight = np.random.randn(in_channel, in_channel)
        q, _ = la.qr(weight)
        w_p, w_l, w_u = la.lu(q.astype(np.float32))
        w_s = np.diag(w_u)
        w_u = np.triu(w_u, 1)
        u_mask = np.triu(np.ones_like(w_u), 1)
        l_mask = u_mask.T

        w_p = torch.from_numpy(w_p)
        w_l = torch.from_numpy(w_l)
        w_s = torch.from_numpy(w_s)
        w_u = torch.from_numpy(w_u)

        self.register_buffer("w_p", w_p)
        self.register_buffer("u_mask", torch.from_numpy(u_mask))
        self.register_buffer("l_mask", torch.from_numpy(l_mask))
        self.register_buffer("s_sign", torch.sign(w_s))
        self.register_buffer("l_eye", torch.eye(l_mask.shape[0]))
        self.w_l = nn.Parameter(w_l)
        self.w_s = nn.Parameter(logabs(w_s))
        self.w_u = nn.Parameter(w_u)

    def forward(self, input):
        _, _, height, width = input.shape

        weight = self.calc_weight()

        out = F.conv2d(input, weight)
        logdet = height * width * torch.sum(self.w_s)

        return out, logdet

    def calc_weight(self):
        weight = (
            self.w_p
            @ (self.w_l * self.l_mask + self.l_eye)
            @ ((self.w_u * self.u_mask) + torch.diag(self.s_sign * torch.exp(self.w_s)))
        )

        return weight.unsqueeze(2).unsqueeze(3)

    def reverse(self, output):
        weight = self.calc_weight()

        return F.conv2d(output, weight.squeeze().inverse().unsqueeze(2).unsqueeze(3))


class Cond_InvConv2dLU(nn.Module):
    def __init__(self, in_channel, mode='conditional', cond_shape=None, inp_shape=None):
        super().__init__()
        self.mode = mode

        # initialize with LU decomposition
        q = la.qr(np.random.randn(in_channel, in_channel))[0].astype(np.float32)
        w_p, w_l, w_u = la.lu(q)

        w_s = np.diag(w_u)  # extract diagonal elements of U into vector w_s
        w_u = np.triu(w_u, 1)  # set diagonal elements of U to 0

        u_mask = np.triu(np.ones_like(w_u), 1)
        l_mask = u_mask.T

        w_p = torch.from_numpy(w_p)
        w_l = torch.from_numpy(w_l)
        w_u = torch.from_numpy(w_u)
        w_s = torch.from_numpy(w_s)

        # non-trainable parameters
        self.register_buffer('w_p', w_p)
        self.register_buffer('u_mask', torch.from_numpy(u_mask))
        self.register_buffer('l_mask', torch.from_numpy(l_mask))
        self.register_buffer('s_sign', torch.sign(w_s))
        self.register_buffer('l_eye', torch.eye(l_mask.shape[0]))

        if self.mode == 'conditional':
            matrices_flattened = torch.cat([torch.flatten(w_l), torch.flatten(w_u), logabs(w_s)])
            self.cond_net = WCondNet(cond_shape, inp_shape, do_lu=True, initial_bias=matrices_flattened)

        else:
            # learnable parameters
            self.w_l = nn.Parameter(w_l)
            self.w_u = nn.Parameter(w_u)
            self.w_s = nn.Parameter(logabs(w_s))

        self.resize = resize(cond_shape)

    def forward(self, inp, condition=None):
        # condition = torch.nn.functional.interpolate(condition, inp.size()[2:], mode='bilinear')
        condition = self.resize(condition)
        _, _, height, width = inp.shape
        weight, s_vector = self.calc_weight(condition)

        out = F.conv2d(inp, weight)
        logdet = height * width * torch.sum(s_vector)
        return out, logdet

    def calc_weight(self, condition=None):
        if self.mode == 'conditional':
            l_matrix, u_matrix, s_vector = self.cond_net(condition)
        else:
            l_matrix, u_matrix, s_vector = self.w_l, self.w_u, self.w_s
    
        weight = (
                self.w_p
                @ (l_matrix * self.l_mask + self.l_eye)  # explicitly make it lower-triangular with 1's on diagonal
                @ ((u_matrix * self.u_mask) + torch.diag(self.s_sign * torch.exp(s_vector)))
        )

        return weight.unsqueeze(2).unsqueeze(3), s_vector


    def reverse_single(self, output, condition=None):
        # condition = torch.nn.functional.interpolate(condition, output.size()[2:], mode='bilinear')
        condition = self.resize(condition)
        weight, _ = self.calc_weight(condition)
        return F.conv2d(output, weight.squeeze().inverse().unsqueeze(2).unsqueeze(3))

    def reverse(self, output, condition=None):
        # condition = torch.nn.functional.interpolate(condition, output.size()[2:], mode='bilinear')
        condition = self.resize(condition)
        batch_size = output.shape[0]
        if batch_size == 1:
            return self.reverse_single(output, condition)
        # reverse one by one for batch size greater than 1. Improving this is not a priority since batch size is usually 1.
        batch_reversed = []
        for i_batch, batch_item in enumerate(output):
            batch_reversed.append(self.reverse(output[i_batch].unsqueeze(0), condition[i_batch].unsqueeze(0)))
        return torch.cat(batch_reversed)

class ZeroConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, padding=1):
        super().__init__()

        self.conv = nn.Conv2d(in_channel, out_channel, 3, padding=0)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()
        self.scale = nn.Parameter(torch.zeros(1, out_channel, 1, 1))

    def forward(self, input):
        out = F.pad(input, [1, 1, 1, 1], value=1)
        out = self.conv(out)
        out = out * torch.exp(self.scale * 3)

        return out


class AffineCoupling(nn.Module):
    def __init__(self, in_channel, filter_size=512, affine=True):
        super().__init__()

        self.affine = affine

        self.net = nn.Sequential(
            nn.Conv2d(in_channel // 2, filter_size, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(filter_size, filter_size, 1),
            nn.ReLU(inplace=True),
            ZeroConv2d(filter_size, in_channel if self.affine else in_channel // 2),
        )

        self.net[0].weight.data.normal_(0, 0.05)
        self.net[0].bias.data.zero_()

        self.net[2].weight.data.normal_(0, 0.05)
        self.net[2].bias.data.zero_()

    def forward(self, input):
        in_a, in_b = input.chunk(2, 1)

        if self.affine:
            log_s, t = self.net(in_a).chunk(2, 1)
            # s = torch.exp(log_s)
            s = F.sigmoid(log_s + 2)
            # out_a = s * in_a + t
            out_b = (in_b + t) * s

            logdet = torch.sum(torch.log(s).view(input.shape[0], -1), 1)

        else:
            net_out = self.net(in_a)
            out_b = in_b + net_out
            logdet = None

        return torch.cat([in_a, out_b], 1), logdet

    def reverse(self, output):
        out_a, out_b = output.chunk(2, 1)

        if self.affine:
            log_s, t = self.net(out_a).chunk(2, 1)
            # s = torch.exp(log_s)
            s = F.sigmoid(log_s + 2)
            # in_a = (out_a - t) / s
            in_b = out_b / s - t

        else:
            net_out = self.net(out_a)
            in_b = out_b - net_out

        return torch.cat([out_a, in_b], 1)




class Cond_AffineCoupling_debugging(nn.Module):
    def __init__(self, in_channel, cond_shape, inp_shape, filter_size=512, affine=True):
        super().__init__()

        self.affine = affine

        self.net_feature = nn.Sequential(
            nn.Conv2d(in_channel, filter_size, 3, padding=1), #nn.Conv2d(in_channel // 2, filter_size, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(filter_size, filter_size, 1),
            nn.ReLU(inplace=True),
            ZeroConv2d(filter_size, in_channel * 2), #ZeroConv2d(filter_size, in_channel if self.affine else in_channel // 2),
        )

        # in_channel=12, cond_shape= (12,64,64)

        self.net_self = nn.Sequential(
            nn.Conv2d(in_channel //2 + cond_shape[0], filter_size, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(filter_size, filter_size, 1),
            nn.ReLU(inplace=True),
            ZeroConv2d(filter_size, in_channel if self.affine else in_channel // 2),
        )

        self.net_feature[0].weight.data.normal_(0, 0.05)
        self.net_feature[0].bias.data.zero_()
        self.net_feature[2].weight.data.normal_(0, 0.05)
        self.net_feature[2].bias.data.zero_()

        self.net_self[0].weight.data.normal_(0, 0.05)
        self.net_self[0].bias.data.zero_()
        self.net_self[2].weight.data.normal_(0, 0.05)
        self.net_self[2].bias.data.zero_()

        self.cond_net = CouplingCondNet(cond_shape, inp_shape)  # without considering batch size dimension

        self.resize = resize(cond_shape)


    def compute_feature_cond(self, cond):
        cond_tensor = self.cond_net(cond)
        inp_a_conditional = cond_tensor
        log_s, t = self.net_feature(cond).chunk(chunks=2, dim=1)
        # print('eel', log_s.max(), log_s.min())
        s = torch.sigmoid(log_s + 2)
        # print('ee', s.max(), s.min())
        return s, t

    def compute_self_cond(self, tensor, cond):
        if cond is not None:  # conditional  # go to here.
            cond_tensor = self.cond_net(cond)
            inp_a_conditional = torch.cat(tensors=[tensor, cond], dim=1)  # concat channel-wise
            log_s, t = self.net_self(inp_a_conditional).chunk(chunks=2, dim=1)
        else:
            log_s, t = self.net_self(tensor).chunk(chunks=2, dim=1)
        # print('dl', log_s.max(), log_s.min())  # log_s diverges
        s = torch.sigmoid(log_s + 2)
        # print('d', s.max(), s.min())
        return s, t

    def forward(self, input, cond):
        # cond = torch.nn.functional.interpolate(cond, input.size()[2:], mode='bilinear')
        cond = self.resize(cond)
        
        # 

        if self.affine:
            # affine injector
            s, t = self.compute_feature_cond(cond)
            out_b = (input + t) * s
            logdet = torch.sum(torch.log(s).view(input.shape[0], -1), 1)
            # affine coupling layer
            out_b1, out_b2 = out_b.chunk(2,1)
            s, t = self.compute_self_cond(out_b1, cond)
            out_b2 = (out_b2 + t) * s
            # print('fff', s.max(), s.min())  # s.min goes to 0.

            logdet = logdet + torch.sum(torch.log(s).view(input.shape[0], -1), 1)

        else:
            in_a, in_b = input.chunk(2, 1)
            net_out = self.net(in_a)
            out_b = in_b + net_out
            logdet = None

        return torch.cat([out_b1, out_b2], 1), logdet

    def reverse(self, output, cond):
        # cond = torch.nn.functional.interpolate(cond, output.size()[2:], mode='bilinear')
        cond = self.resize(cond)
        # out_a, out_b = output.chunk(2, 1)

        if self.affine:
            out_b1, out_b2 = output.chunk(2,1)
            s, t = self.compute_self_cond(out_b1, cond)
            out_b2 = out_b2 / s - t 
            out_b = torch.cat([out_b1, out_b2], dim=1)

            s, t = self.compute_feature_cond(cond)
            out_b = out_b / s - t

        else:
            out_a, out_b = output.chunk(2, 1)
            net_out = self.net(out_a)
            in_b = out_b - net_out

        return out_b

class Cond_AffineCoupling_original(nn.Module):
    def __init__(self, in_channel, cond_shape, inp_shape, filter_size=512, affine=True):
        super().__init__()

        self.affine = affine

        self.net = nn.Sequential(
            nn.Conv2d(in_channel // 2 + cond_shape[0], filter_size, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(filter_size, filter_size, 1),
            nn.ReLU(inplace=True),
            ZeroConv2d(filter_size, in_channel if self.affine else in_channel // 2),
        )

        self.net[0].weight.data.normal_(0, 0.05)
        self.net[0].bias.data.zero_()

        self.net[2].weight.data.normal_(0, 0.05)
        self.net[2].bias.data.zero_()

        self.cond_net = CouplingCondNet(cond_shape, inp_shape)  # without considering batch size dimension
        self.resize = resize(cond_shape)

    def compute_coupling_params(self, tensor, cond):
        if cond is not None:  # conditional
            # cond_tensor = self.cond_net(cond) # removing conditional network (CN)
            inp_a_conditional = torch.cat(tensors=[tensor, cond], dim=1)  # concat channel-wise
            log_s, t = self.net(inp_a_conditional).chunk(chunks=2, dim=1)
        else:
            log_s, t = self.net(tensor).chunk(chunks=2, dim=1)
        s = torch.sigmoid(log_s + 2)
        return s, t

    def forward(self, input, cond):
        # cond = torch.nn.functional.interpolate(cond, input.size()[2:], mode='bilinear')
        cond = self.resize(cond)
        
        in_a, in_b = input.chunk(2, 1)

        if self.affine:
            s, t = self.compute_coupling_params(in_a, cond)
            out_b = (in_b + t) * s
            logdet = torch.sum(torch.log(s).view(input.shape[0], -1), 1)

        else:
            net_out = self.net(in_a)
            out_b = in_b + net_out
            logdet = None

        return torch.cat([in_a, out_b], 1), logdet

    def reverse(self, output, cond):
        # cond = torch.nn.functional.interpolate(cond, output.size()[2:], mode='bilinear')
        cond = self.resize(cond)
        out_a, out_b = output.chunk(2, 1)

        if self.affine:
            s, t = self.compute_coupling_params(out_a, cond)
            in_b = out_b / s - t

        else:
            net_out = self.net(out_a)
            in_b = out_b - net_out

        return torch.cat([out_a, in_b], 1)


class Flow(nn.Module):
    def __init__(self, in_channel, affine=True, conv_lu=True):
        super().__init__()

        self.actnorm = ActNorm(in_channel)

        if conv_lu:
            self.invconv = InvConv2dLU(in_channel)

        else:
            self.invconv = InvConv2d(in_channel)

        self.coupling = AffineCoupling(in_channel, affine=affine)

    def forward(self, input):
        act_out, logdet = self.actnorm(input)
        w_out, det1 = self.invconv(act_out)
        out, det2 = self.coupling(w_out)

        logdet = logdet + det1
        if det2 is not None:
            logdet = logdet + det2

        # return out, logdet
        return act_out, w_out, out, logdet

    def reverse(self, output):
        input = self.coupling.reverse(output)
        input = self.invconv.reverse(input)
        input = self.actnorm.reverse(input)

        return input


class transition(nn.Module):
    def __init__(self, in_channel, conv_lu=True):
        super().__init__()

        self.actnorm = ActNorm(in_channel)

        if conv_lu:
            self.invconv = InvConv2dLU(in_channel)

        else:
            self.invconv = InvConv2d(in_channel)

    def forward(self, input):
        act_out, act_logdet = self.actnorm(input)
        w_out, w_det1 = self.invconv(act_out)

        logdet = act_logdet + w_det1

        out = w_out

        return act_out, w_out, out, logdet

    def reverse(self, output):
        input = self.invconv.reverse(output)
        input = self.actnorm.reverse(input)

        return input


class Cond_Flow(nn.Module):
    def __init__(self, in_channel, cond_shape, inp_shape, affine=True, conv_lu=True):
        super().__init__()
        self.actnorm = ActNorm(in_channel) # in_channel : 12

        if conv_lu:
            self.invconv = InvConv2dLU(in_channel)
        else:
            self.invconv = InvConv2d(in_channel)

        self.coupling = Cond_AffineCoupling_original(in_channel, cond_shape, inp_shape, affine=affine)
        # self.coupling = Cond_AffineCoupling_debugging(in_channel, cond_shape, inp_shape, affine=affine)

    def forward(self, input, conditions):
        # act_out, act_logdet = self.actnorm(input, conditions['act_cond'])
        # w_out, w_det1 = self.invconv(act_out, conditions['w_cond'])
        act_out, act_logdet = self.actnorm(input)
        w_out, w_det1 = self.invconv(act_out)
        out, det2 = self.coupling(w_out, conditions['coupling_cond'])

        logdet = act_logdet + w_det1
        if det2 is not None:
            logdet = logdet + det2

        return act_out, w_out, out, logdet

    def reverse(self, output, conditions):
        # input = self.coupling.reverse(output, conditions['coupling_cond'])
        # input = self.invconv.reverse(input, conditions['w_cond'])
        input = self.coupling.reverse(output,conditions['coupling_cond'])
        input = self.invconv.reverse(input)
        input = self.actnorm.reverse(input)

        return input

def gaussian_log_p(x, mean, log_sd):
    return -0.5 * log(2 * pi) - log_sd - 0.5 * (x - mean) ** 2 / torch.exp(2 * log_sd)


def gaussian_sample(eps, mean, log_sd):
    return mean + torch.exp(log_sd) * eps


class Block(nn.Module):
    def __init__(self, in_channel, n_flow, split=True, affine=True, conv_lu=True):
        super().__init__()

        squeeze_dim = in_channel * 4

        self.flows = nn.ModuleList()
        self.flows.append(transition(squeeze_dim, conv_lu=conv_lu))
        for i in range(n_flow):
            self.flows.append(Flow(squeeze_dim, affine=affine, conv_lu=conv_lu))

        self.split = split

        if split:
            self.prior = ZeroConv2d(in_channel * 2, in_channel * 4)

        else:
            self.prior = ZeroConv2d(in_channel * 4, in_channel * 8)

    def forward(self, input):
        b_size, n_channel, height, width = input.shape
        squeezed = input.view(b_size, n_channel, height // 2, 2, width // 2, 2)
        squeezed = squeezed.permute(0, 1, 3, 5, 2, 4)
        out = squeezed.contiguous().view(b_size, n_channel * 4, height // 2, width // 2)

        logdet = 0

        flows_outs = []  # list of tensor, each element of which is the output of the corresponding flow step
        w_outs = []
        act_outs = []

        total_log_det = 0

        for _, flow in enumerate(self.flows):
            flow_output = flow(out)
            flow_output = to_dict('flow', flow_output)
            out, log_det = flow_output['out'], flow_output['log_det']
            total_log_det = total_log_det + log_det

            # appending flow_outs - done by the left glow
            flows_outs.append(out)

            # appending w_outs - done by the left glow
            w_out = flow_output['w_out']
            w_outs.append(w_out)

            # appending act_outs - done by the left glow
            act_out = flow_output['act_out']
            act_outs.append(act_out)

        if self.split:
            out, z_new = out.chunk(2, 1)
            mean, log_sd = self.prior(out).chunk(2, 1)
            log_p = gaussian_log_p(z_new, mean, log_sd)
            log_p = log_p.view(b_size, -1).sum(1)

        else:
            zero = torch.zeros_like(out)
            mean, log_sd = self.prior(zero).chunk(2, 1)
            log_p = gaussian_log_p(out, mean, log_sd)
            log_p = log_p.view(b_size, -1).sum(1)
            z_new = out

        # return out, logdet, log_p, z_new
        return {
            'act_outs': act_outs,
            'w_outs': w_outs,
            'flows_outs': flows_outs,
            'out': out,
            'total_log_det': total_log_det,
            'log_p': log_p,
            'z_new': z_new
        }

    def reverse(self, output, eps=None, reconstruct=False):
        input = output

        if reconstruct:
            if self.split:
                input = torch.cat([output, eps], 1)

            else:
                input = eps

        else:
            if self.split:
                mean, log_sd = self.prior(input).chunk(2, 1)
                z = gaussian_sample(eps, mean, log_sd)
                input = torch.cat([output, z], 1)

            else:
                zero = torch.zeros_like(input)
                # zero = F.pad(zero, [1, 1, 1, 1], value=1)
                mean, log_sd = self.prior(zero).chunk(2, 1)
                z = gaussian_sample(eps, mean, log_sd)
                input = z

        for flow in self.flows[::-1]:
            input = flow.reverse(input)

        b_size, n_channel, height, width = input.shape

        unsqueezed = input.view(b_size, n_channel // 4, 2, 2, height, width)
        unsqueezed = unsqueezed.permute(0, 1, 4, 2, 5, 3)
        unsqueezed = unsqueezed.contiguous().view(
            b_size, n_channel // 4, height * 2, width * 2
        )

        return unsqueezed


class Cond_Block(nn.Module):
    def __init__(self, in_channel, n_flow, inp_shape, cond_shape, split=True, affine=True, conv_lu=True):
        super().__init__()

        squeeze_dim = in_channel * 4

        self.flows = nn.ModuleList()
        self.flows.append(transition(squeeze_dim, conv_lu=conv_lu)) # here is the transition layer!
        for i in range(n_flow):
            self.flows.append(Cond_Flow(squeeze_dim, inp_shape, cond_shape, affine=affine, conv_lu=conv_lu))

        self.split = split

        if split:
            self.prior = ZeroConv2d(in_channel * 2, in_channel * 4)

        else:
            self.prior = ZeroConv2d(in_channel * 4, in_channel * 8)

    def forward(self, input, conditions):
        b_size, n_channel, height, width = input.shape
        squeezed = input.view(b_size, n_channel, height // 2, 2, width // 2, 2)
        squeezed = squeezed.permute(0, 1, 3, 5, 2, 4)
        out = squeezed.contiguous().view(b_size, n_channel * 4, height // 2, width // 2)

        logdet = 0

        flows_outs = []  # list of tensor, each element of which is the output of the corresponding flow step
        w_outs = []
        act_outs = []

        total_log_det = 0
        for i, flow in enumerate(self.flows):
            if i>0:
                act_cond, w_cond, coupling_cond = extract_conds(conditions, i-1) # It should be modified if left and right glow are equivalent
                condition = {'act_cond' : act_cond, 'w_cond' : w_cond, 'coupling_cond' : coupling_cond}
                flow_output = flow(out, condition)
            else:
                flow_output = flow(out)

            flow_output = to_dict('flow', flow_output)
            out, log_det = flow_output['out'], flow_output['log_det']
            total_log_det = total_log_det + log_det

            # appending flow_outs - done by the left glow
            flows_outs.append(out)

            # appending w_outs - done by the left glow
            w_out = flow_output['w_out']
            w_outs.append(w_out)

            # appending act_outs - done by the left glow
            act_out = flow_output['act_out']
            act_outs.append(act_out)

        if self.split:

            out, z_new = out.chunk(2, 1)
            mean, log_sd = self.prior(out).chunk(2, 1)
            log_p = gaussian_log_p(z_new, mean, log_sd)
            log_p = log_p.view(b_size, -1).sum(1)

        else:
            zero = torch.zeros_like(out)
            mean, log_sd = self.prior(zero).chunk(2, 1)
            log_p = gaussian_log_p(out, mean, log_sd)
            log_p = log_p.view(b_size, -1).sum(1)
            z_new = out

        # return out, logdet, log_p, z_new
        return {
            'act_outs': act_outs,
            'w_outs': w_outs,
            'flows_outs': flows_outs,
            'out': out,
            'total_log_det': total_log_det,
            'log_p': log_p,
            'z_new': z_new
        }

    def reverse(self, output, conditions, eps=None, reconstruct=False):
        input = output

        if reconstruct:
            if self.split:
                input = torch.cat([output, eps], 1)

            else:
                input = eps

        else:
            if self.split:
                mean, log_sd = self.prior(input).chunk(2, 1)
                z = gaussian_sample(eps, mean, log_sd)
                input = torch.cat([output, z], 1)

            else:
                zero = torch.zeros_like(input)
                # zero = F.pad(zero, [1, 1, 1, 1], value=1)
                mean, log_sd = self.prior(zero).chunk(2, 1)
                z = gaussian_sample(eps, mean, log_sd)
                input = z
        for i, flow in enumerate(self.flows[::-1]):
            if i != len(self.flows) - 1:
                act_cond, w_cond, coupling_cond = extract_conds(conditions, i)
                conds = make_cond_dict(act_cond, w_cond, coupling_cond)
                input = flow.reverse(input, conds)
            else:
                input = flow.reverse(input)

        b_size, n_channel, height, width = input.shape

        unsqueezed = input.view(b_size, n_channel // 4, 2, 2, height, width)
        unsqueezed = unsqueezed.permute(0, 1, 4, 2, 5, 3)
        unsqueezed = unsqueezed.contiguous().view(
            b_size, n_channel // 4, height * 2, width * 2
        )

        return unsqueezed


class Glow(nn.Module):
    def __init__(
        self, in_channel, n_flow, n_block, affine=True, conv_lu=True
    ):
        super().__init__()

        self.blocks = nn.ModuleList()
        n_channel = in_channel
        for i in range(n_block - 1):
            self.blocks.append(Block(n_channel, n_flow, affine=affine, conv_lu=conv_lu))
            n_channel *= 2
        self.blocks.append(Block(n_channel, n_flow, split=False, affine=affine))

    def forward(self, input):
        log_p_sum = 0
        log_det = 0
        out = input
        z_outs = []

        all_flows_outs = []  # a 2d list, each element of which corresponds to the flows_outs of each Block
        all_w_outs = []  # 2d list
        all_act_outs = []  # 2d list


        for block in self.blocks:
            # out, det, log_p, z_new = block(out)
            block_out = block(out)
            
            out, det, log_p = block_out['out'], block_out['total_log_det'], block_out['log_p']
            z_new = block_out['z_new']

            # appending flows_outs - done by the left_glow
            flows_out = block_out['flows_outs']
            all_flows_outs.append(flows_out)

            # appending w_outs - done by the left_glow
            w_outs = block_out['w_outs']
            all_w_outs.append(w_outs)

            # appending act_outs - done by the left_glow
            act_outs = block_out['act_outs']
            all_act_outs.append(act_outs)

            z_outs.append(z_new)
            log_det = log_det + det

            if log_p is not None:
                log_p_sum = log_p_sum + log_p

        # return log_p_sum, logdet, z_outs
        return {
            'all_act_outs': all_act_outs,
            'all_w_outs': all_w_outs,
            'all_flows_outs': all_flows_outs,
            'z_outs': z_outs,
            'log_p_sum': log_p_sum,
            'log_det': log_det
        }

    def reverse(self, z_list, reconstruct=False):
        for i, block in enumerate(self.blocks[::-1]):
            if i == 0:
                input = block.reverse(z_list[-1], z_list[-1], reconstruct=reconstruct)

            else:
                input = block.reverse(input, z_list[-(i + 1)], reconstruct=reconstruct)

        return input


class Cond_Glow(nn.Module):
    def __init__(
        self, in_channel, n_flow, n_block, input_shapes, cond_shapes, affine=True, conv_lu=True
    ):
        super().__init__()
        self.n_blocks = n_block
        self.blocks = nn.ModuleList()
        n_channel = in_channel
        for i in range(n_block - 1):
            inp_shape = input_shapes[i]
            cond_shape = cond_shapes[i]

            self.blocks.append(Cond_Block(n_channel, n_flow, inp_shape, cond_shape, affine=affine, conv_lu=conv_lu))
            n_channel *= 2
        self.blocks.append(Cond_Block(n_channel, n_flow, input_shapes[n_block - 1], cond_shapes[n_block - 1], split=False, affine=affine))

    def forward(self, inp, left_glow_out):
        conditions = self.prep_conds(left_glow_out, direction='forward')
        log_p_sum = 0
        log_det = 0
        out = inp
        z_outs = []

        all_flows_outs = []  # a 2d list, each element of which corresponds to the flows_outs of each Block
        all_w_outs = []  # 2d list
        all_act_outs = []  # 2d list

        for i, block in enumerate(self.blocks):
            act_cond, w_cond, coupling_cond = extract_conds(conditions, i)
            conds = make_cond_dict(act_cond, w_cond, coupling_cond)

            block_out = block(out, conds)

            out, det, log_p = block_out['out'], block_out['total_log_det'], block_out['log_p']
            z_new = block_out['z_new']

            # appending flows_outs - done by the left_glow
            flows_out = block_out['flows_outs']
            all_flows_outs.append(flows_out)

            # appending w_outs - done by the left_glow
            w_outs = block_out['w_outs']
            all_w_outs.append(w_outs)

            # appending act_outs - done by the left_glow
            act_outs = block_out['act_outs']
            all_act_outs.append(act_outs)

            z_outs.append(z_new)
            log_det = log_det + det
            log_p_sum = log_p_sum + log_p

        return {
            'all_act_outs': all_act_outs,
            'all_w_outs': all_w_outs,
            'all_flows_outs': all_flows_outs,
            'z_outs': z_outs,
            'log_p_sum': log_p_sum,
            'log_det': log_det
        }

    def reverse(self, z_list, reconstruct=False, left_glow_out=None):
        conditions = self.prep_conds(left_glow_out, direction='reverse')
        inp = None
        rec_list = [reconstruct] * self.n_blocks  # make a list of True or False

        # Block reverse operations one by one
        for i, block in enumerate(self.blocks[::-1]):  # it starts from the last Block
            act_cond, w_cond, coupling_cond = extract_conds(conditions, i)
            conds = make_cond_dict(act_cond, w_cond, coupling_cond)

            reverse_input = z_list[-1] if i == 0 else inp
            block_reverse = block.reverse(output=reverse_input,  # Block reverse operation
                                          eps=z_list[-(i + 1)],
                                          reconstruct=rec_list[-(i + 1)],
                                          conditions=conds)
            inp = block_reverse
        return inp

    def prep_conds(self, left_glow_out, direction):
        act_cond = left_glow_out['all_act_outs']
        w_cond = left_glow_out['all_w_outs']  # left_glow_out in the forward direction
        coupling_cond = left_glow_out['all_flows_outs']
        coupling_cond_new = []
        for c in coupling_cond:
            coupling_cond_new.append(c[1:])
        coupling_cond = coupling_cond_new

        # make conds a dictionary
        conditions = make_cond_dict(act_cond, w_cond, coupling_cond)

        # reverse lists for reverse operation
        if direction == 'reverse':
            conditions['act_cond'] = [list(reversed(cond)) for cond in list(reversed(conditions['act_cond']))]  # reverse 2d list
            conditions['w_cond'] = [list(reversed(cond)) for cond in list(reversed(conditions['w_cond']))]
            conditions['coupling_cond'] = [list(reversed(cond)) for cond in list(reversed(conditions['coupling_cond']))]
        return conditions


class Cond_Glow_Skip(nn.Module):
    def __init__(
        self, in_channel, n_flow, n_block, input_shapes, cond_shapes, affine=True, conv_lu=True
    ):
        super().__init__()
        self.n_blocks = n_block
        self.blocks = nn.ModuleList()
        n_channel = in_channel
        for i in range(n_block - 1):
            inp_shape = input_shapes[i]
            cond_shape = cond_shapes[i]

            self.blocks.append(Cond_Block(n_channel, n_flow, inp_shape, cond_shape, affine=affine, conv_lu=conv_lu))
            n_channel *= 2
        self.blocks.append(Cond_Block(n_channel, n_flow, input_shapes[n_block - 1], cond_shapes[n_block - 1], split=False, affine=affine))

    def forward(self, inp, left_glow_out, image_lr_resize):
        conditions = self.prep_conds(left_glow_out, direction='forward')
        log_p_sum = 0
        log_det = 0
        out = inp
        z_outs = []

        all_flows_outs = []  # a 2d list, each element of which corresponds to the flows_outs of each Block
        all_w_outs = []  # 2d list
        all_act_outs = []  # 2d list

        for i, block in enumerate(self.blocks):
            act_cond, w_cond, coupling_cond = extract_conds(conditions, i)
            conds = make_cond_dict(act_cond, w_cond, coupling_cond)

            # subtract image_lr_resize squeezes
            out = out - image_lr_resize
            block_out = block(out, conds)

            # image_lr_resize squeeze for next image_lr_resize
            b_size, n_channel, height, width = image_lr_resize.size()
            squeezed = image_lr_resize.view(b_size, n_channel, height // 2, 2, width // 2, 2)
            squeezed = squeezed.permute(0, 1, 3, 5, 2, 4)
            image_lr_resize = squeezed.contiguous().view(b_size, n_channel * 4, height // 2, width // 2)
            image_lr_resize, _ = image_lr_resize.chunk(2, 1)

            out, det, log_p = block_out['out'], block_out['total_log_det'], block_out['log_p']
            z_new = block_out['z_new']

            # appending flows_outs - done by the left_glow
            flows_out = block_out['flows_outs']
            all_flows_outs.append(flows_out)

            # appending w_outs - done by the left_glow
            w_outs = block_out['w_outs']
            all_w_outs.append(w_outs)

            # appending act_outs - done by the left_glow
            act_outs = block_out['act_outs']
            all_act_outs.append(act_outs)

            z_outs.append(z_new)
            log_det = log_det + det
            log_p_sum = log_p_sum + log_p

        return {
            'all_act_outs': all_act_outs,
            'all_w_outs': all_w_outs,
            'all_flows_outs': all_flows_outs,
            'z_outs': z_outs,
            'log_p_sum': log_p_sum,
            'log_det': log_det
        }

    def reverse(self, z_list, reconstruct=False, left_glow_out=None, image_lr_resize=None):
        conditions = self.prep_conds(left_glow_out, direction='reverse')
        inp = None
        rec_list = [reconstruct] * self.n_blocks  # make a list of True or False

        # generate image_lr_resize squeezes
        image_lr_resize_list = [image_lr_resize]
        for i in range(self.n_blocks-1):
            b_size, n_channel, height, width = image_lr_resize.size()
            squeezed = image_lr_resize.view(b_size, n_channel, height // 2, 2, width // 2, 2)
            squeezed = squeezed.permute(0, 1, 3, 5, 2, 4)
            image_lr_resize = squeezed.contiguous().view(b_size, n_channel * 4, height // 2, width // 2)
            image_lr_resize, _ = image_lr_resize.chunk(2, 1)
            image_lr_resize_list.append(image_lr_resize)

        # Block reverse operations one by one
        for i, block in enumerate(self.blocks[::-1]):  # it starts from the last Block
            act_cond, w_cond, coupling_cond = extract_conds(conditions, i)
            conds = make_cond_dict(act_cond, w_cond, coupling_cond)

            reverse_input = z_list[-1] if i == 0 else inp
            block_reverse = block.reverse(output=reverse_input,  # Block reverse operation
                                          eps=z_list[-(i + 1)],
                                          reconstruct=rec_list[-(i + 1)],
                                          conditions=conds)
            inp = block_reverse + image_lr_resize_list[-1-i]
        return inp

    def prep_conds(self, left_glow_out, direction):
        act_cond = left_glow_out['all_act_outs']
        w_cond = left_glow_out['all_w_outs']  # left_glow_out in the forward direction
        coupling_cond = left_glow_out['all_flows_outs']
        coupling_cond_new = []
        for c in coupling_cond:
            coupling_cond_new.append(c[1:])
        coupling_cond = coupling_cond_new

        # make conds a dictionary
        conditions = make_cond_dict(act_cond, w_cond, coupling_cond)

        # reverse lists for reverse operation
        if direction == 'reverse':
            conditions['act_cond'] = [list(reversed(cond)) for cond in list(reversed(conditions['act_cond']))]  # reverse 2d list
            conditions['w_cond'] = [list(reversed(cond)) for cond in list(reversed(conditions['w_cond']))]
            conditions['coupling_cond'] = [list(reversed(cond)) for cond in list(reversed(conditions['coupling_cond']))]
        return conditions
