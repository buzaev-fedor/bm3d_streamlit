import numpy as np
from scipy.linalg import hadamard
import math


def ht_filtering_hadamard(group_3D, sigma, lambda_hard_3D, do_weight):  # group_3D shape=(n*n, nSx_r)
    """
    :hard threshold filtering after hadamard transform
    :param group_3D:
    :param sigma:
    :param lambda_hard_3D:
    :param do_weight:
    :return:
    """
    nSx_r = group_3D.shape[-1]
    coef_norm = math.sqrt(nSx_r)

    group_3D_h = hadamard_transform(group_3D)

    # hard threshold filtering in this block
    threshold = lambda_hard_3D * sigma * coef_norm
    weight = np.sum(np.where(np.abs(group_3D_h) > threshold, 1, 0))
    group_3D_h = np.where(np.abs(group_3D_h) > threshold, group_3D_h, 0.)

    group_3D = hadamard_transform(group_3D_h)
    group_3D *= 1.0 / nSx_r
    if do_weight:
        weight = 1. / (sigma * sigma * weight) if weight > 0. else 1.

    return group_3D, weight


def hadamard_transform(vec):
    n = vec.shape[-1]
    h_mat = hadamard(n).astype(np.float64)
    return vec @ h_mat
