# -*- coding: utf-8 -*-
# @Organization  : Alibaba XR-Lab, CUHK-SZ
# @Author        : Lingteng Qiu
# @Email         : 220019047@link.cuhk.edu.cn
# @Time          : 2024-01-23 11:21:30
# @Function      : An example to compute variance metrics of normal prediction.

import argparse
import csv
import glob
import os
import time
from collections import defaultdict

import cv2
import numpy as np
import torch

DATASET = {
    "DIODE": "NormalDiffusion_eval/DIODE",
}

import multiprocessing


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


def worker(gt_result, ref_image, cur_state_list, high_frequency=False):

    angles = []
    rmses = []

    normal_gt = cv2.imread(gt_result)
    ref_image = cv2.imread(ref_image)
    normal_gt = normal_gt / 255 * 2 - 1

    # normal_gt = cv2.resize(normal_gt, (512, 512))

    normal_gt_norm = np.linalg.norm(normal_gt, axis=-1)
    fg_mask_gt = (normal_gt_norm > 0.5) & (normal_gt_norm < 1.5)

    if high_frequency:

        edges = cv2.Canny(ref_image, 0, 50)
        kernel = np.ones((3, 3), np.uint8)
        fg_mask_gt = cv2.dilate(edges, kernel, iterations=1) / 255
        fg_mask_gt = edges / 255
        fg_mask_gt = fg_mask_gt == 1.0

    angles = []
    for target in cur_state_list:

        normal_pred = cv2.imread(target)
        normal_pred = cv2.resize(normal_pred, (normal_gt.shape[1], normal_gt.shape[0]))
        normal_pred = normal_pred / 255 * 2 - 1

        normal_pred_norm = np.linalg.norm(normal_pred, axis=-1)
        normal_pred = tqlo.safe_normalize(normal_pred)

        # fg_mask_pred = (normal_pred_norm > 0.5) & (normal_pred_norm < 1.5)
        # fg_mask = fg_mask_gt & fg_mask_pred

        fg_mask = fg_mask_gt
        dot_product = (normal_pred * normal_gt).sum(axis=-1)
        dot_product = np.clip(dot_product, -1, 1)
        dot_product = dot_product[fg_mask]

        angle = np.arccos(dot_product) / np.pi * 180

        angle = angle.mean().item()

        angles.append(angle)

    print(f"processing {gt_result}")

    return angles


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--input", "-i", required=True, type=str)
    parser.add_argument("--model_name", "-m", type=str, default="geowizard")
    parser.add_argument("--hf", action="store_true", help="high frequency error map")

    opt = parser.parse_args()
    save_metric_path = "./eval_results/metrics_variance/{opt.model_name}"

    save_path = strip(opt.input)
    model_name = save_path.split("/")[-2]
    sampling_name = os.path.basename(save_path)

    root_dir = f"{opt.input}"

    seed_model_list = sorted(
        glob.glob(os.path.join(opt.input, f"{opt.model_name}_seed*"))
    )
    # seed_model_list = sorted(glob.glob(os.path.join(opt.input, f'seed*')))
    seed_model_list = [
        is_img(sorted(glob.glob(os.path.join(seed_model_path, "*.png"))))
        for seed_model_path in seed_model_list
    ]

    seed_states_list = []

    length = None
    for seed_idx, seed_model in enumerate(seed_model_list):
        data_states = obtain_states(seed_model)
        gt_results = data_states.pop("gt")
        ref_results = data_states.pop("ref")

        keys = data_states.keys()
        last_key = sorted(keys, key=lambda x: int(x.replace("step", "")))[-1]

        try:
            if length is None:
                length = len(data_states[last_key])
            else:
                assert length == len(data_states[last_key]), print(seed_idx)
        except:
            continue

        seed_states_list.append(data_states[last_key])

    num_cpus = multiprocessing.cpu_count()

    states = data_states.keys()

    start = time.time()

    print(f"using cpu: {num_cpus}")

    pool = multiprocessing.Pool(processes=num_cpus)
    metrics_results = []

    for idx, gt_result in enumerate(gt_results):
        ref_result = ref_results[idx]

        cur_seed_states = [
            seed_states_list[_][idx] for _ in range(len(seed_states_list))
        ]

        metrics_results.append(
            pool.apply_async(worker, (gt_result, ref_result, cur_seed_states, opt.hf))
        )

    pool.close()
    pool.join()

    times = time.time() - start
    print(f"All processes completed using time {times:.4f} s...")

    metrics_results = [metrics_result.get() for metrics_result in metrics_results]

    metrics_results = np.asarray(metrics_results)

    print("*" * 10)
    print("variance: {}".format(metrics_results.var(axis=-1).mean()))

    print("*" * 10)
