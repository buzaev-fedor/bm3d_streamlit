import numpy as np


def image2patches_naive(image, patch_height, patch_weight):
    """
    :cut the image into patches
    :param image:
    :param patch_height:
    :param patch_weight:
    :return:
    """
    height, weight = image.shape[0], image.shape[1]
    patch_table = np.zeros((height - patch_height + 1, weight - patch_weight + 1, patch_height, patch_weight), dtype=np.float64)
    for i in range(height - patch_height + 1):
        for j in range(weight - patch_weight + 1):
            patch_table[i][j] = image[i:i + patch_height, j:j + patch_weight]

    return patch_table


def image2patches(image, patch_height, patch_weight):
    """
    :cut the image into patches
    :param image:
    :param patch_height:
    :param patch_weight:
    :return:
    """
    image_height, image_weight = image.shape[0], image.shape[1]
    image_height_idx = np.arange(image_height - patch_height + 1)[:, np.newaxis, np.newaxis, np.newaxis]

    image_weight_idx = np.arange(image_weight - patch_weight + 1)[np.newaxis, :, np.newaxis, np.newaxis]

    patch_height_idx = np.arange(patch_height)[np.newaxis, np.newaxis, :, np.newaxis]
    patch_weight_idx = np.arange(patch_weight)[np.newaxis, np.newaxis, np.newaxis, :]

    height_idx = image_height_idx + patch_height_idx
    weight_idx = image_weight_idx + patch_weight_idx

    return image[height_idx, weight_idx].astype(np.float64)
