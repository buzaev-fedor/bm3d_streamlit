import numpy as np

from utils import ind_initialize, sd_weighting, build_3D_group
from precompute_BM import precompute_BM
from bior_2d import bior_2d_forward, bior_2d_reverse
from dct_2d import dct_2d_forward, dct_2d_reverse
from image_to_patches import image2patches
from ht_filtering_hadamard import ht_filtering_hadamard


def bm3d_1st_step(sigma, image_noisy, n_Hard, k_Hard, N_Hard, p_Hard, lambda_Hard_3D, tau_Match, use_SD, tau_2D):
    """
    :param sigma:
    :param image_noisy:
    :param n_Hard:
    :param k_Hard:  кол-во патчей, которые попадают в куб
    :param N_Hard:
    :param p_Hard:
    :param lambda_Hard_3D:
    :param tau_Match:
    :param use_SD:
    :param tau_2D:
    :return:
    """
    height, width = image_noisy.shape[0], image_noisy.shape[1]

    row_ind = ind_initialize(height - k_Hard + 1, n_Hard, p_Hard)
    column_ind = ind_initialize(width - k_Hard + 1, n_Hard, p_Hard)

    ri_rj_N__ni_nj, threshold_count = precompute_BM(image_noisy, length_side_patch=k_Hard, number_patches=N_Hard, length_side_area=n_Hard, tau_match=tau_Match)
    group_len = int(np.sum(threshold_count))
    group_3D_table = np.zeros((group_len, k_Hard, k_Hard))
    weight_table = np.zeros((height, width))

    all_patches = image2patches(image_noisy, k_Hard, k_Hard)
    if tau_2D == 'DCT':
        fre_all_patches = dct_2d_forward(all_patches)
    else:  # 'BIOR'
        fre_all_patches = bior_2d_forward(all_patches)

    acc_pointer = 0
    for i_r in row_ind:
        for j_r in column_ind:
            nSx_r = threshold_count[i_r, j_r]
            group_3D = build_3D_group(fre_all_patches, ri_rj_N__ni_nj[i_r, j_r], nSx_r)
            group_3D, weight = ht_filtering_hadamard(group_3D, sigma, lambda_Hard_3D, not use_SD)
            group_3D = group_3D.transpose((2, 0, 1))
            group_3D_table[acc_pointer:acc_pointer + nSx_r] = group_3D
            acc_pointer += nSx_r

            if use_SD:
                weight = sd_weighting(group_3D)

            weight_table[i_r, j_r] = weight

    if tau_2D == 'DCT':
        group_3D_table = dct_2d_reverse(group_3D_table)
    else:  # 'BIOR'
        group_3D_table = bior_2d_reverse(group_3D_table)

    # aggregation part
    numerator = np.zeros_like(image_noisy, dtype=np.float64)

    denominator = np.pad(np.zeros((image_noisy.shape[0] - 2 * n_Hard, image_noisy.shape[1] - 2 * n_Hard),
                                  dtype=np.float64), n_Hard, 'constant', constant_values=1.)
    acc_pointer = 0

    for i_r in row_ind:
        for j_r in column_ind:
            nSx_r = threshold_count[i_r, j_r]
            N_ni_nj = ri_rj_N__ni_nj[i_r, j_r]
            group_3D = group_3D_table[acc_pointer:acc_pointer + nSx_r]
            acc_pointer += nSx_r
            weight = weight_table[i_r, j_r]
            for n in range(nSx_r):
                ni, nj = N_ni_nj[n]
                patch = group_3D[n]

                # numerator[ni:ni + k_Hard, nj:nj + k_Hard] += patch * kaiser_window * weight
                # denominator[ni:ni + k_Hard, nj:nj + k_Hard] += kaiser_window * weight
                numerator[ni:ni + k_Hard, nj:nj + k_Hard] += patch * weight
                denominator[ni:ni + k_Hard, nj:nj + k_Hard] += weight

    return numerator / denominator

