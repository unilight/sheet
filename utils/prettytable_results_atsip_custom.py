#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2024 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

"""Print results within one checkpoint with prettytable."""

import argparse
import csv
import sys
import os
from prettytable import PrettyTable, MARKDOWN
import numpy as np

train_sets = [
    ["bvcc", "hubert_large_ll60k"],
    ["somos", "hubert_large_ll60k"],
    ["singmos", "wav2vec2_large_ll60k"],
    ["nisqa", "wavlm_large"], 
    ["pstn", "wav2vec2_large_ll60k"], 
    ["tencent", "wavlm_large"], 
    ["tmhint-qi", "hubert_large_ll60k"],
    ["bvcc+nisqa+pstn+singmos+somos+tencent+tmhint-qi", "wavlm_large"]
]

synthetic_sets = [
    "bc19_test",
    "bvcc_test",
    "singmos_test",
    "somos_test",
    "vmc23_track1a_test",
    "vmc23_track1b_test",
    "vmc23_track2_test"
]

non_sets = [
    "nisqa_FOR",
    "nisqa_LIVETALK",
    "nisqa_P501",
    "tmhintqi_test",
    "vmc23_track3_test"
]

order = [
    "bvcc_test",
    "somos_test",
    "singmos_test",
    "bc19_test",
    "vmc23_track1a_test",
    "vmc23_track1b_test",
    "vmc23_track2_test",
    "nisqa_FOR",
    "nisqa_LIVETALK",
    "nisqa_P501",
    "tmhintqi_test",
    "vmc23_track3_test"
]

templates = [
    "/home/z44476r/data/Experiments/sheet/egs/{}/exp/ssl-mos-{}-1337/results/checkpoint-best",
    "/home/z44476r/data/Experiments/sheet/egs/{}/exp/ssl-mos-u-{}-1337/results/checkpoint-best",
    "/home/z44476r/data/Experiments/sheet/egs/{}/exp/ssl-mos-u-lnll-{}-1337/results/checkpoint-best",
]


def get_ccs(log_filepath, _type, test_set):
    ccs = {}
    with open(log_filepath, "r") as f:
        lines = f.read().splitlines()
        for line in lines:
            if "[UTT]" in line:

                # [UTT][ MSE = 1.337 | LCC = 0.220 | SRCC = 0.161 ] [SYS][ MSE = 1.257 | LCC = 0.3571 | SRCC = 0.2867 ]
                line = (
                    line.replace("|", "")
                    .replace("[", "")
                    .replace("]", "")
                    .replace("=", "")
                    .split()
                )
                if _type == "cc":
                    ccs[d] = float(line[-10])
                    # if d in synthetic_sets:
                    #     ccs[d] = float(line[-1])
                    # elif d in non_sets:
                    #     ccs[d] = float(line[-10])
                elif _type == "mse":
                    if test_set in synthetic_sets:
                        ccs[test_set] = float(line[-5])
                    elif test_set in non_sets:
                        ccs[test_set] = float(line[-12])

    return ccs


def gaussian_nll(y, mu, log_var):
    var = np.exp(log_var)
    nll = 0.5 * (np.log(2 * np.pi * var) + (y - mu)**2 / var)
    return nll  # per-sample


def laplace_nll(y, mu, log_b):
    b = np.exp(log_b)
    nll = np.log(2 * b) + np.abs(y - mu) / b
    return nll  # per-sample


def get_nlls(results_filepath, loss, test_set):
    nlls = {}
    with open(results_filepath, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        contents = [line for line in reader]

    if loss == "gnll":
        calc_fn = gaussian_nll
    elif loss == "lnll":
        calc_fn = laplace_nll
    nlls[test_set] = np.mean([
        calc_fn(
            float(line["avg_score"]),
            float(line["answer"]),
            float(line["logvar"]),
        )
        for line in contents]
    )

    return nlls


def main():
    """Run training process."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--type",
        type=str,
        default="cc",
        choices=["cc", "mse"],
    )
    args = parser.parse_args()

    for train_set, ssl_model in train_sets:
        for template in templates:
            _dir = template.format(train_set, ssl_model)
            if not os.path.exists(_dir):
                continue
            
            for test_set in sorted(os.listdir(_dir)):
                if test_set not in order:
                    continue

                log_filepath = os.path.join(_dir, test_set, "inference.log")
                if os.path.isfile(log_filepath):
                    ccs = get_ccs(log_filepath, args.type, test_set)

                if "-u-" in template or "-lnll-" in template:
                    if "-lnll-" in template:
                        loss = "lnll"
                    elif "-u-" in template:
                        loss = "gnll"
                    results_filepath = os.path.join(args.dir, d, "results.csv")
                    if os.path.isfile(results_filepath):
                        nlls = get_nlls(results_filepath, loss, test_set)

                    
            print(train_set, " ".join(f"{ccs[d]:.3f}" for d in order), f"{np.mean(list(ccs.values())):.3f}")

if __name__ == "__main__":
    main()
