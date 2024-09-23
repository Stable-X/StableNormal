# -*- coding: utf-8 -*-
# @Organization  : Alibaba XR-Lab, CUHK-SZ
# @Author        : Lingteng Qiu
# @Email         : 220019047@link.cuhk.edu.cn
# @Time          : 2024-01-23 11:21:30
# @Function      : An example to compute metrics of normal prediction.


import argparse
import csv
import multiprocessing
import os
import time
from collections import defaultdict

import cv2
import numpy as np
import torch


def dot(x, y):
    """dot product (along the last dim).

    Args:
        x (Union[Tensor, ndarray]): x, [..., C]
        y (Union[Tensor, ndarray]): y, [..., C]

    Returns:
        Union[Tensor, ndarray]: x dot y, [..., 1]
    """
    if isinstance(x, np.ndarray):
        return np.sum(x * y, -1, keepdims=True)
    else:
        return torch.sum(x * y, -1, keepdim=True)


def is_format(f, format):
    """if a file's extension is in a set of format

    Args:
        f (str): file name.
        format (Sequence[str]): set of extensions (both '.jpg' or 'jpg' is ok).

    Returns:
        bool: if the file's extension is in the set.
    """
    ext = os.path.splitext(f)[1].lower()  # include the dot
    return ext in format or ext[1:] in format


def is_img(input_list):
    return list(filter(lambda x: is_format(x, [".jpg", ".jpeg", ".png"]), input_list))


def length(x, eps=1e-20):
    """length of an array (along the last dim).

    Args:
        x (Union[Tensor, ndarray]): x, [..., C]
        eps (float, optional): eps. Defaults to 1e-20.

    Returns:
        Union[Tensor, ndarray]: length, [..., 1]
    """
    if isinstance(x, np.ndarray):
        return np.sqrt(np.maximum(np.sum(x * x, axis=-1, keepdims=True), eps))
    else:
        return torch.sqrt(torch.clamp(dot(x, x), min=eps))


def safe_normalize(x, eps=1e-20):
    """normalize an array (along the last dim).

    Args:
        x (Union[Tensor, ndarray]): x, [..., C]
        eps (float, optional): eps. Defaults to 1e-20.

    Returns:
        Union[Tensor, ndarray]: normalized x, [..., C]
    """

    return x / length(x, eps)


def strip(s):
    if s[-1] == "/":
        return s[:-1]
    else:
        return s


def obtain_states(img_list):
    all_states = defaultdict(list)
    for img in img_list:
        states = os.path.basename(img)
        states = os.path.splitext(states)[0].split("_")[-1]

        all_states[states].append(img)

    for key in all_states.keys():
        all_states[key] = sorted(all_states[key])

    return all_states


def writer_csv(filename, data):
    with open(filename, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(data)


def worker(gt_result, cur_state_list):

    angles = []
    rmses = []

    normal_gt = cv2.imread(gt_result)
    normal_gt = normal_gt / 255 * 2 - 1

    normal_gt_norm = np.linalg.norm(normal_gt, axis=-1)

    for target in cur_state_list:

        normal_pred = cv2.imread(target)
        normal_pred = cv2.resize(normal_pred, (normal_gt.shape[1], normal_gt.shape[0]))
        normal_pred = normal_pred / 255 * 2 - 1

        normal_pred_norm = np.linalg.norm(normal_pred, axis=-1)
        normal_pred = safe_normalize(normal_pred)

        fg_mask_pred = (normal_pred_norm > 0.5) & (normal_pred_norm < 1.5)
        fg_mask_gt = (normal_gt_norm > 0.5) & (normal_gt_norm < 1.5)

        # fg_mask = fg_mask_gt & fg_mask_pred
        fg_mask = fg_mask_gt

        rmse = np.sqrt(((normal_pred - normal_gt) ** 2)[fg_mask].sum(axis=-1).mean())
        dot_product = (normal_pred * normal_gt).sum(axis=-1)

        dot_product = np.clip(dot_product, -1, 1)
        dot_product = dot_product[fg_mask]

        angle = np.arccos(dot_product) / np.pi * 180

        # Create an error map visualization
        error_map = np.zeros_like(normal_gt[:, :, 0])
        error_map[fg_mask] = angle
        error_map = np.clip(
            error_map, 0, 90
        )  # Clipping the values to [0, 90] for better visualization
        error_map = cv2.applyColorMap(np.uint8(error_map * 255 / 90), cv2.COLORMAP_JET)

        # Save the error map
        # cv2.imwrite(f"{root_dir}/{os.path.basename(source).replace('_gt.png', f'_{method}_error.png')}", error_map)

        angles.append(angle)
        rmses.append(rmse.item())

    print(f"processing {gt_result}")

    return gt_result, angles, rmses


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--dataset_name", default="DIODE", type=str, choices=["DIODE"])
    parser.add_argument("--input", "-i", required=True, type=str)

    save_metric_path = "./eval_results/metrics"

    opt = parser.parse_args()

    save_path = strip(opt.input)
    model_name = save_path.split("/")[-2]
    sampling_name = os.path.basename(save_path)

    root_dir = f"{opt.input}"
    save_metric_path = os.path.join(save_metric_path, f"{model_name}_{sampling_name}")

    os.makedirs(save_metric_path, exist_ok=True)

    img_list = [os.path.join(root_dir, x) for x in os.listdir(root_dir)]
    img_list = is_img(img_list)

    data_states = obtain_states(img_list)

    gt_results = data_states.pop("gt")
    ref_results = data_states.pop("ref")

    num_cpus = multiprocessing.cpu_count()

    states = data_states.keys()
    states = sorted(states, key=lambda x: int(x.replace("step", "")))

    start = time.time()

    print(f"using cpu: {num_cpus}")

    pool = multiprocessing.Pool(processes=num_cpus)
    metrics_results = []

    for idx, gt_result in enumerate(gt_results):
        cur_state_list = [data_states[state][idx] for state in states]
        metrics_results.append(pool.apply_async(worker, (gt_result, cur_state_list)))

    pool.close()
    pool.join()

    times = time.time() - start
    print(f"All processes completed using time {times:.4f} s...")

    metrics_results = [metrics_result.get() for metrics_result in metrics_results]

    angles_csv = [["name", *states]]
    rmse_csv = [["name", *states]]

    angle_arr = []
    rmse_arr = []

    for metrics in metrics_results:
        name, angle, rmse = metrics

        angles_csv.append([name, *angle])

        angle_arr.append(angle)

    print(angles_csv[0])

    tokens = [[] for _ in range(len(angles_csv[0]))]

    for angles in angles_csv[1:]:
        for token_idx, angle in enumerate(angles):
            tokens[token_idx].append(angle)

    new_tokens = [[] for _ in range(len(angles_csv[0]))]
    for token_idx, token in enumerate(tokens):

        if token_idx == 0:
            new_tokens[token_idx] = np.asarray(token)
        else:
            new_tokens[token_idx] = np.concatenate(token)

    for i in range(1, len(new_tokens)):
        angle_arr = new_tokens[i]

        pct_gt_5 = 100.0 * np.sum(angle_arr < 11.25, axis=0) / angle_arr.shape[0]
        pct_gt_10 = 100.0 * np.sum(angle_arr < 22.5, axis=0) / angle_arr.shape[0]
        pct_gt_30 = 100.0 * np.sum(angle_arr < 30, axis=0) / angle_arr.shape[0]
        media = np.median(angle_arr)
        mean = np.mean(angle_arr)

        print("*" * 10)
        print(("{:.3f}\t" * 5).format(mean, media, pct_gt_5, pct_gt_10, pct_gt_30))
        print("*" * 10)
