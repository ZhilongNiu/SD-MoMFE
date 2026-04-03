
import numpy as np
from torch.autograd import Variable
import argparse
import datetime
import time
import logging
import os.path as osp
import os
import torch
from torch.utils.data import DataLoader
import warnings
from PIL import Image
from torch import Tensor
from typing import Tuple, Union, List

def gray2rgb(img_ir_batch: torch.Tensor) -> torch.Tensor:
    R = img_ir_batch
    G = img_ir_batch
    B = img_ir_batch
    return torch.cat([R, G, B], dim=1)


def RGB2YCrCb(input_im: Tensor) -> Tensor:

    device = input_im.device
    im_flat = input_im.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
    
    R = im_flat[:, 0]
    G = im_flat[:, 1]
    B = im_flat[:, 2]
    
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5
    
    Y = torch.unsqueeze(Y, 1)
    Cr = torch.unsqueeze(Cr, 1)
    Cb = torch.unsqueeze(Cb, 1)
    temp = torch.cat((Y, Cr, Cb), dim=1).to(device)
    
    out = (
        temp.reshape(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,
        )
        .transpose(1, 3)
        .transpose(2, 3)
    )
    return out

def YCrCb2RGB(input_im: Tensor) -> Tensor:

    device = input_im.device
    
    im_flat = input_im.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
    
    mat = torch.tensor(
        [[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
    ).to(device)
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).to(device)
    
    temp = (im_flat + bias).mm(mat).to(device)
    
    out = (
        temp.reshape(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,
        )
        .transpose(1, 3)
        .transpose(2, 3)
    )
    return out

def tensor2img(img: Tensor, is_norm: bool = True) -> np.ndarray:

    img = img.cpu().float().numpy()
    
    if img.shape[0] == 1:
        img = np.tile(img, (3, 1, 1))
    
    if is_norm:
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
    
    img = np.transpose(img, (1, 2, 0)) * 255.0
    return img.astype(np.uint8)

def save_img_single(img: Tensor, name: str, is_norm: bool = True) -> None:

    img = tensor2img(img, is_norm=is_norm)
    img = Image.fromarray(img)
    img.save(name)

def save_img_single_resize(img: Tensor, name: str, shape: Tuple[int, int], 
                         is_norm: bool = True) -> None:

    img = tensor2img(img, is_norm=is_norm)
    img = Image.fromarray(img)
    img = img.resize(shape, Image.LANCZOS)
    img.save(name)

