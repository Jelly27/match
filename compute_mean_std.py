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
            file_path = os.path.join(root, file)
            relative_path = os.path.relpath(file_path, root_dir)
            relative_paths.append(os.path.join(root_dir, relative_path))
    return relative_paths


def main():
    start = time()
    path = "XR_SHOULDER"

    img_name_list = get_relative_paths(path)[1:]
    cumulative_mean = 0
    cumulative_std = 0
    for img_name in tqdm(img_name_list):
        img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
        mean = np.mean(img)
        std = np.std(img)
        cumulative_mean += mean
        cumulative_std += std

    mean = cumulative_mean / len(img_name_list)
    std = cumulative_std / len(img_name_list)
    print(f"mean: {mean}, {mean / 255}")
    print(f"std: {std}, {std / 255}")
    print(f"用时: {timedelta(seconds=int(time() - start))}")
    """
        mean: 131.89993813429433, 0.5172546593501739
        std: 58.96875804769272, 0.2312500315595793
        用时: 0:00:52
        [0.5173, 0.2313]

        去除多类 4113张
        mean: 132.65929430324667, 0.5202325266793987
        std: 58.8832943322999, 0.2309148797345094
        用时: 0:00:44
        
        XR_SHOULDER 2分类
        mean: 65.8149335691691, 0.2580977787026239
        std: 33.37347416732489, 0.13087636928362703
        用时: 0:00:22
    """


if __name__ == '__main__':
    main()