import numpy as np
from scipy.linalg import hadamard


def wiener_filtering_hadamard(group_3D_img, group_3D_est, sigma, do_weight):
    """
    :wiener_filtering after hadamard transform
    :param group_3D_img:
    :param group_3D_est:
    :param sigma:
    :param do_weight:
    :return:
    """
    assert group_3D_img.shape == group_3D_est.shape

    coef = 1.0 / group_3D_img.shape[-1]

    group_3D_img_h = hadamard_transform(group_3D_img)  # along nSx_r axis
    group_3D_est_h = hadamard_transform(group_3D_est)

    # wiener filtering in this block
    value = np.power(group_3D_est_h, 2) * coef
    value /= (value + sigma ** 2)
    weight = np.sum(value)

    group_3D_est = hadamard_transform(group_3D_img_h * value * coef)

    if do_weight:
        weight = 1. / (sigma * sigma * weight) if weight > 0. else 1.

    return group_3D_est, weight


def hadamard_transform(vector):
    return vector @ hadamard(vector.shape[-1]).astype(np.float64)
