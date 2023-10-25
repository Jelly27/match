import os
import cv2
import numpy as np
from tqdm import tqdm
from datetime import timedelta
from time import time


def get_relative_paths(root_dir):
    relative_paths = []
    for root, directories, files in os.walk(root_dir):
        for file in files:
            if file[0] == ".":
                continue
            file_path = os.path.join(root, file)
            relative_path = os.path.relpath(file_path, root_dir)
            relative_paths.append(os.path.join(root_dir, relative_path))
    return relative_paths


def main(path: str, mode: int = 1):
    """
        计算图像的mean和std
    :param path:
    :param mode: 0灰度，1RGB，-1透明通道
    :return:
    """
    start = time()

    img_name_list = get_relative_paths(path)
    cumulative_mean = 0
    cumulative_std = 0
    if mode == 0:
        axis = None
    elif mode == 1:
        axis = (0, 1)
    else:
        raise "mode error"
    for img_name in tqdm(img_name_list):
        img = cv2.imread(img_name, mode)
        mean = np.mean(img, axis)
        std = np.std(img, axis)
        cumulative_mean += mean
        cumulative_std += std

    mean = cumulative_mean / len(img_name_list) / 256
    std = cumulative_std / len(img_name_list) / 256
    print(f"mean: {np.round(mean, 5)}")
    print(f"std: {np.round(std, 5)}")
    print(f"用时: {timedelta(seconds=int(time() - start))}")


if __name__ == '__main__':
    main("blood_cell")
    """
        blood_cell 4分类 用时52''
        灰度 mean=[0.6546], std=[0.2571]
        RGB [0.65791 0.63881 0.67597], [0.25579 0.25846 0.25891]
    """
