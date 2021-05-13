# coding=utf-8
import cv2
import numpy as np
from collections import Counter
import time
from numba import jit

@jit(nopython=True)
def calcIJ(img_patch):
    total_p = img_patch.shape[1]
    assert total_p % 2, "error window size"
    center_p = img_patch[:, total_p // 2]
    mean_p = (np.sum(img_patch, axis=1) - center_p) / (total_p - 1)
    return list(map(lambda x, y: (x, y), center_p, mean_p))


def calcEntropy2dSpeedUp(img, win_w=3, win_h=3):
    height = img.shape[0]

    ext_x = int(win_w / 2)
    ext_y = int(win_h / 2)

    ext_h_part = np.zeros([height, ext_x], img.dtype)
    tem_img = np.hstack((ext_h_part, img, ext_h_part))
    ext_v_part = np.zeros([ext_y, tem_img.shape[1]], img.dtype)
    final_img = np.vstack((ext_v_part, tem_img, ext_v_part))

    new_width = final_img.shape[1]
    new_height = final_img.shape[0]
    patch = np.zeros((img.shape[0] * img.shape[1], win_w * win_h))
    for i in range(win_h):
        for j in range(win_w):
            patch[:, i * win_h + j] = final_img[i : new_height - win_h + 1 + i, j : new_width - win_w + 1 + j].flatten()
    IJ = calcIJ(patch)
    Fij = Counter(IJ).items()
    return np.sum(list(map(lambda x: -x * (np.log(x) / np.log(2)), map(lambda item: item[1] * 1.0 / (new_height * new_width), Fij))))

if __name__ == '__main__':
    img1 = cv2.imread("./imgs/00000.jpg", cv2.IMREAD_GRAYSCALE)
    t1 = time.time()
    H1 = calcEntropy2dSpeedUp(img1, 3, 3)
    t2 = time.time()
    print(H1)
    print(t2 - t1, 's')
