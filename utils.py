import cv2
import numpy as np
from PIL import Image
import math


def add_padding(image, N):
    return np.pad(image, ((N, N), (N, N)), 'symmetric')


def add_gaussian_noise(image, sigma, seed=None):
    if seed is not None:
        np.random.seed(seed)
    image = image + (sigma * np.random.randn(*image.shape)).astype(np.int)
    image = np.clip(image, 0., 255., out=None)
    image = image.astype(np.uint8)
    return image

def ind_initialize(max_size, N, step):
    ind = range(N, max_size - N, step)
    if ind[-1] < max_size - N - 1:
        ind = np.append(ind, np.array([max_size - N - 1]), axis=0)
    return ind


def get_kaiserWindow(kHW):
    k = np.kaiser(kHW, 2)
    k_2d = k[:, np.newaxis] @ k[np.newaxis, :]
    return k_2d


def get_coef(kHW):
    coef_norm = np.zeros(kHW * kHW)
    coef_norm_inv = np.zeros(kHW * kHW)
    coef = 0.5 / (float(kHW))
    for i in range(kHW):
        for j in range(kHW):
            if i == 0 and j == 0:
                coef_norm[i * kHW + j] = 0.5 * coef
                coef_norm_inv[i * kHW + j] = 2.0
            elif i * j == 0:
                coef_norm[i * kHW + j] = 0.7071067811865475 * coef
                coef_norm_inv[i * kHW + j] = 1.414213562373095
            else:
                coef_norm[i * kHW + j] = 1.0 * coef
                coef_norm_inv[i * kHW + j] = 1.0

    return coef_norm, coef_norm_inv


def sd_weighting(group_3D):
    N = group_3D.size

    mean = np.sum(group_3D)
    std = np.sum(group_3D * group_3D)

    res = (std - mean * mean / N) / (N - 1)
    weight = 1.0 / np.sqrt(res) if res > 0. else 0.
    return weight


def array_to_image(array):
    array = np.clip(array, 0., 255.).astype(np.uint8)
    return Image.fromarray(array)


def compute_rmse_psnr(image_1, image_2):
    image_1 = image_1.astype(np.float64) / 255.
    image_2 = image_2.astype(np.float64) / 255.
    rmse = np.sqrt(np.mean((image_1 - image_2) ** 2))
    if rmse == 0:
        return "Same Image"

    return rmse, 20 * math.log10(1. / rmse)


def build_3D_group(fre_all_patches, N__ni_nj, nSx_r):
    """
    :stack frequency patches into a 3D block
    :param fre_all_patches: all frequency patches
    :param N__ni_nj: the position of the N most similar patches
    :param nSx_r: how many similar patches according to threshold
    :return: the 3D block
    """
    _, _, k, k_ = fre_all_patches.shape
    assert k == k_
    group_3D = np.zeros((nSx_r, k, k))
    for n in range(nSx_r):
        ni, nj = N__ni_nj[n]
        group_3D[n, :, :] = fre_all_patches[ni, nj]
    group_3D = group_3D.transpose((1, 2, 0))
    return group_3D
