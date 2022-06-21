from utils import add_padding
from bm3d_first_step import bm3d_1st_step
from bm3d_second_step import bm3d_2nd_step
import numpy as np


def run_bm3d(noisy_image, sigma,
             n_hard, k_hard, p_hard, tau_hard, use_SD_hard, tau_2D_hard, lambda_3D_hard,
             n_wien, k_wien, p_wiener, tau_wien, use_SD_wien, tau_2D_wien):

    padded_noised_image = add_padding(noisy_image, n_hard)
    image_first_step = bm3d_1st_step(sigma, padded_noised_image, n_hard, k_hard, n_hard, p_hard, lambda_3D_hard,
                                     tau_hard, use_SD_hard, tau_2D_hard)
    # remove padding
    image_first_step = image_first_step[n_hard: -n_hard, n_hard: -n_hard]

    assert not np.any(np.isnan(image_first_step))
    image_first_padded = add_padding(image_first_step, n_wien)
    padded_noised_image = add_padding(noisy_image, n_wien)
    img_denoised = bm3d_2nd_step(sigma, padded_noised_image, image_first_padded, n_wien, k_wien,
                                 n_wien, p_wiener, tau_wien, use_SD_wien, tau_2D_wien)
    img_denoised = img_denoised[n_wien: -n_wien, n_wien: -n_wien]

    return image_first_step, img_denoised

