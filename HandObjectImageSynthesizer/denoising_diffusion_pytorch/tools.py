import torch
from torchvision import utils
import matplotlib.pyplot as plt
import numpy as np
import os
import imageio
import cv2


def make_gif(img1, img2, exp_name, name):
    if not os.path.exists(f'gif_results/{exp_name}'):
        os.mkdir(f'gif_results/{exp_name}')
    img1, img2 = cv2.imread(img1), cv2.imread(img2)
    with imageio.get_writer(f'gif_results/{exp_name}/{name}.gif', mode='I', duration=0.5) as writer:
        writer.append_data(img1)
        writer.append_data(img2)


def tensor_gpu(batch, device=torch.cuda, check_on=True):

    def check_on_gpu(tensor_):
        if isinstance(tensor_, str) or isinstance(tensor_, list):
            tensor_g = tensor_
        else:
            tensor_g = tensor_.to(device)
        return tensor_g

    def check_off_gpu(tensor_):
        if isinstance(tensor_, str) or isinstance(tensor_, list):
            return tensor_

        if tensor_.is_cuda:
            tensor_c = tensor_.cpu()
        else:
            tensor_c = tensor_
        tensor_c = tensor_c.detach().numpy()
        return tensor_c

    if torch.cuda.is_available():
        if check_on:
            for k, v in batch.items():
                batch[k] = check_on_gpu(v)
        else:
            for k, v in batch.items():
                batch[k] = check_off_gpu(v)
    else:
        if check_on:
            batch = batch
        else:
            for k, v in batch.items():
                batch[k] = v.detach().numpy()

    return batch


def save_depth(
    tensor,
    fp,
    nrow=8,
    padding=2,
    normalize=True,
    range=None,
    scale_each=False,
    pad_value=0,
    format=None,
) -> None:
    '''Save a given Tensor into an image file.

    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        fp (string or file object): A filename or a file object
        format(Optional):  If omitted, the format to use is determined from the filename extension.
            If a file object was used instead of a filename, this parameter should always be used.
        **kwargs: Other arguments are documented in ``make_grid``.
    '''
    from PIL import Image
    grid = utils.make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value, normalize=normalize, range=range, scale_each=scale_each)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    ndarr = plt.cm.plasma(ndarr[:, :, 0])
    ndarr = np.clip(ndarr * 255 + 0.5, 0, 255).astype(np.uint8)
    im = Image.fromarray(ndarr)
    im.save(fp, format=format)
