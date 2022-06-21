import numpy as np


def precompute_BM(image, length_side_patch, number_patches, length_side_area, tau_match):
    """
    :search for similar patches
    :param image: input image
    :param length_side_patch: length of side of patch
    :param number_patches: how many patches are stacked
    :param length_side_area: length of side of search area
    :param tau_match: threshold determine whether two patches are similar
    :return top_n_similar_patches: The top N most similar patches to the referred patch
    :return threshold_count: according to tau_match how many patches are similar to the referred one
    """
    image = image.astype(np.float64)
    height, width = image.shape
    num_sides_area = 2 * length_side_area + 1
    threshold = tau_match * length_side_patch * length_side_patch
    sum_table = np.ones((num_sides_area, num_sides_area, height, width)) * 2 * threshold  # di, dj, ph, pw
    row_add_mat, column_add_mat = get_add_patch_matrix(height, width, length_side_area, length_side_patch)
    diff_margin = np.pad(np.ones((height - 2 * length_side_area, width - 2 * length_side_area)), length_side_area, 'constant', constant_values=0.)
    sum_margin = (1 - diff_margin) * 2 * threshold

    for di in range(-length_side_area, length_side_area + 1):
        for dj in range(-length_side_area, length_side_area + 1):
            t_img = translation_2d_mat(image, right=-dj, down=-di)
            diff_table_2 = (image - t_img) * (image - t_img) * diff_margin

            sum_diff_2 = row_add_mat @ diff_table_2 @ column_add_mat
            sum_table[di + length_side_area, dj + length_side_area] = np.maximum(sum_diff_2, sum_margin)  # sum_table (2n+1, 2n+1, height, width)

    sum_table = sum_table.reshape((num_sides_area * num_sides_area, height * width))  # di_dj, ph_pw
    sum_table_T = sum_table.transpose((1, 0))  # ph_pw__di_dj
    argsort = np.argpartition(sum_table_T, range(number_patches))[:, :number_patches]
    argsort[:, 0] = (num_sides_area * num_sides_area - 1) // 2
    argsort_di = argsort // num_sides_area - length_side_area
    argsort_dj = argsort % num_sides_area - length_side_area
    near_pi = argsort_di.reshape((height, width, -1)) + np.arange(height)[:, np.newaxis, np.newaxis]
    near_pj = argsort_dj.reshape((height, width, -1)) + np.arange(width)[np.newaxis, :, np.newaxis]
    top_n_similar_patches = np.concatenate((near_pi[:, :, :, np.newaxis], near_pj[:, :, :, np.newaxis]), axis=-1)

    sum_filter = np.where(sum_table_T < threshold, 1, 0)
    threshold_count = np.sum(sum_filter, axis=1)
    threshold_count = closest_power_of_2(threshold_count, max_=number_patches)
    threshold_count = threshold_count.reshape((height, width))

    return top_n_similar_patches, threshold_count


def get_add_patch_matrix(h, w, nHW, kHW):
    row_add = np.pad(np.eye(h - 2 * nHW), nHW, 'constant')
    row_add_mat = row_add.copy()
    for k in range(1, kHW):
        row_add_mat += translation_2d_mat(row_add, right=k, down=0)

    column_add = np.pad(np.eye(w - 2 * nHW), nHW, 'constant')
    column_add_mat = column_add.copy()
    for k in range(1, kHW):
        column_add_mat += translation_2d_mat(column_add, right=0, down=k)

    return row_add_mat, column_add_mat


def translation_2d_mat(mat, right, down):
    mat = np.roll(mat, right, axis=1)
    mat = np.roll(mat, down, axis=0)
    return mat


def closest_power_of_2(M, max_):
    M = np.where(max_ < M, max_, M)
    while max_ > 1:
        M = np.where((max_ // 2 < M) * (M < max_), max_ // 2, M)
        max_ //= 2
    return M
