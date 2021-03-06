import torch
import numpy as np
import nvidia_smi

# from globals import device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_edges(t):
    """
    This function is taken from: https://github.com/NVIDIA/pix2pixHD.
    :param t:
    :return:
    """
    edge = torch.cuda.ByteTensor(t.size()).zero_()
    # comparing with the left pixels
    edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1]).type(torch.uint8)
    # comparing with the right pixels
    edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1]).type(torch.uint8)
    # comparing with the lower pixels
    edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :]).type(torch.uint8)
    # comparing with upper  pixels
    edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :]).type(torch.uint8)
    return edge.float()


def label_to_tensor(label, height, width, count=0):
    if count == 0:
        arr = np.zeros((10, height, width))
        arr[label] = 1

    else:
        arr = np.zeros((count, 10, height, width))
        arr[:, label, :, :] = 1

    return torch.from_numpy(arr.astype(np.float32))


def save_checkpoint(path_to_save, optim_step, model, optimizer, loss, lr):
    name = path_to_save + f'/optim_step={optim_step}.pt'
    checkpoint = {'loss': loss,
                  'lr': lr,
                  'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict()}

    torch.save(checkpoint, name)
    print(f'In [save_checkpoint]: save state dict done at: "{name}"')


def load_checkpoint(path_to_load, optim_step, model, optimizer, resume_train=True):
    name = path_to_load + f'/optim_step={optim_step}.pt'
    checkpoint = torch.load(name, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    loss = checkpoint['loss']
    lr = checkpoint['lr'] if 'lr' in checkpoint.keys() else None  # backward compatibility for checkpoints that do not store lr

    print(f'In [load_checkpoint]: load state dict done from: "{name}"')

    # putting the model in the correct mode
    if resume_train:
        model.train()
    else:
        model.eval()
        for param in model.parameters():  # freezing the layers when using only for evaluation
            param.requires_grad = False

    if optimizer is not None:
        return model.to(device), optimizer, loss, lr
    return model.to(device), None, loss, lr


def show_memory_usage():
    nvidia_smi.nvmlInit()
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)  # GPU number
    mem_res = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    # print('=' * 50)
    # print(f'mem: {mem_res.used / (1024 ** 3)} (GiB)')  # usage in GiB
    print(f'mem usage: {100 * (mem_res.used / mem_res.total):.3f}%')  # percentage
    # print('=' * 50)


def tensor_size(tensor):
    return tensor.element_size() * tensor.nelement()
