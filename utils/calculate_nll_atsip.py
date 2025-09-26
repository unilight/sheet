#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2024 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

"""Print results within one checkpoint with prettytable."""

import argparse
import csv
import sys
import os
import numpy as np
from prettytable import PrettyTable, MARKDOWN

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

def gaussian_nll(y, mu, log_var):
    var = np.exp(log_var)
    nll = 0.5 * (np.log(2 * np.pi * var) + (y - mu)**2 / var)
    return nll  # per-sample

def laplace_nll(y, mu, log_b):
    b = np.exp(log_b)
    nll = np.log(2 * b) + np.abs(y - mu) / b
    return nll  # per-sample

def main():
    """Run training process."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dir",
        type=str,
    )
    parser.add_argument(
        "--loss",
        type=str,
        required=True,
        choices=["gnll", "lnll"],
    )
    parser.add_argument(
        "--type",
        type=str,
        default="cc",
        choices=["cc", "mse"],
    )
    args = parser.parse_args()

    nlls = {}

    for d in sorted(os.listdir(args.dir)):
        results_filepath = os.path.join(args.dir, d, "results.csv")
        if os.path.isfile(results_filepath):
            with open(results_filepath, newline="") as csvfile:
                reader = csv.DictReader(csvfile)
                contents = [line for line in reader]

            if args.loss == "gnll":
                calc_fn = gaussian_nll
            elif args.loss == "lnll":
                calc_fn = laplace_nll
            nlls[d] = np.mean([
                calc_fn(
                    float(line["avg_score"]),
                    float(line["answer"]),
                    float(line["logvar"]),
                )
                for line in contents]
            )

    print(" ".join(f"{nlls[d]:.3f}" for d in order))

if __name__ == "__main__":
    main()
