import numpy as np
import torch
import numpy as np
import gymnasium as gym
import os
from collections import deque
import random
from torch.utils.data import Dataset
from torch import nn

class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def make_dir(dir_path):
    try:
        os.makedirs(dir_path, exist_ok=True)
    except OSError:
        pass
    return dir_path

def random_crop(images, output_size=112):
    """
    args:
    images: np.array shape (B,C,H,W)
    output_size: output size, assuming square images
    returns: np.array
    """
    n, c, h, w = images.shape
    crop_max = h - output_size + 1
    w1 = np.random.randint(0, crop_max, n)
    h1 = np.random.randint(0, crop_max, n)
    cropped = np.empty((n, c, output_size, output_size), dtype=images.dtype)
    for i, (image, w11, h11) in enumerate(zip(images, w1, h1)):
        cropped[i] = image[:, h11 : h11 + output_size, w11 : w11 + output_size]
    return cropped


def center_crop(image, output_size=112):
    h, w = image.shape[1:]
    assert h >= output_size
    new_h, new_w = output_size, output_size

    top = (h - new_h) // 2
    left = (w - new_w) // 2

    image = image[:, top : top + new_h, left : left + new_w]
    return image


def batch_center_crop(images, output_size=112):
    n, c, h, w = images.shape
    cropped = np.empty((n, c, output_size, output_size), dtype=images.dtype)
    for i, image in enumerate(images):
        cropped[i] = center_crop(image, output_size)
    return cropped


def no_aug(x):
    return x
