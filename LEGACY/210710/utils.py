from torch import nn
import torch


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

def calc_inp_shapes(n_channels, image_size, n_blocks):
    # calculates z shapes (inputs) after SQUEEZE operation (before Block operations) - e.g. channels: 12, 24, 48, 96
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

def make_cond_dict(act_cond, w_cond, coupling_cond):
    return {'act_cond': act_cond, 'w_cond': w_cond, 'coupling_cond': coupling_cond}


def extract_conds(conditions, level):
    act_cond = conditions['act_cond'][level]
    w_cond = conditions['w_cond'][level]
    coupling_cond = conditions['coupling_cond'][level]
    return act_cond, w_cond, coupling_cond


def to_dict(module, output):
    assert module == 'flow'
    return {
        'act_out': output[0],
        'w_out': output[1],
        'out': output[2],
        'log_det': output[3]
    }


