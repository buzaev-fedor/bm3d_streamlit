import math
import pywt
import numpy as np


def bior_2d_forward(image):
    """
    :wavelet forward transform
    :param bior_img:
    :return:
    """
    assert image.shape[-1] == image.shape[-2]
    iter_max = int(math.log2(image.shape[-1]))

    coeffs = pywt.wavedec2(image, 'bior1.5', level=iter_max, mode='periodization')
    wave_image = np.zeros_like(image, dtype=np.float64)

    N = 1
    wave_image[..., :N, :N] = coeffs[0]
    for i in range(1, iter_max + 1):
        wave_image[..., N:2 * N, N:2 * N] = coeffs[i][2]
        wave_image[..., 0:N, N: 2 * N] = -coeffs[i][1]
        wave_image[..., N: 2 * N, 0:N] = -coeffs[i][0]
        N *= 2
    return wave_image


def bior_2d_reverse(bior_image):
    """
    :wavelet reverse transform
    :param bior_image:
    :return:
    """
    assert bior_image.shape[-1] == bior_image.shape[-2]
    iter_max = int(math.log2(bior_image.shape[-1]))

    N = 1
    rec_coeffs = [bior_image[..., 0:1, 0:1]]
    for i in range(iter_max):
        LL = bior_image[..., N:2 * N, N:2 * N]
        HL = -bior_image[..., 0:N, N: 2 * N]
        LH = -bior_image[..., N: 2 * N, 0:N]
        t = (LH, HL, LL)
        rec_coeffs.append(t)
        N *= 2

    rec_im = pywt.waverec2(rec_coeffs, 'bior1.5', mode='periodization')
    return rec_im
