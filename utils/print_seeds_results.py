#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2024 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

"""

Print results across three seeds with prettytable.
This is for the MOS-Bench journal paper.

"""

import argparse
import csv
import sys
import os
import numpy as np
from prettytable import PrettyTable, MARKDOWN

seeds = ["2337", "3337", "4337"]
checkpoint_name = "checkpoint-best"
# checkpoint_name = "np_checkpoint-best/naive_knn"
# checkpoint_name = "np_checkpoint-best/domain_id_knn_1"
test_sets = [
    "bvcc_test",
    "somos_test",
    "singmos_test",
    "nisqa_FOR",
    "nisqa_LIVETALK",
    "nisqa_P501",
    "tmhintqi_test",
    "bc19_test",
    "vmc23_track1a_test",
    "vmc23_track1b_test",
    "vmc23_track2_test",
    "vmc23_track3_test",
]

def main():
    """Run training process."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tag",
        required=True,
        type=str,
        help=("exp tag name, not including the seed"),
    )
    args = parser.parse_args()

    table = PrettyTable()
    table.set_style(MARKDOWN)
    table.field_names = [
        "dataset",
        "Utt MSE",
        "Utt LCC",
        "Utt SRCC",
        "Sys MSE",
        "Sys LCC",
        "Sys SRCC",
    ]
    table.align = "r"
    table.align["dataset"] = "l"
    table.float_format = ".3"

    string = []

    for test_set in test_sets:
        utt_mse, utt_lcc, utt_srcc, sys_mse, sys_lcc, sys_srcc = [], [], [], [], [], []
        for seed in seeds:
            log_filepath = os.path.join("exp", f"{args.tag}-{seed}", "results", checkpoint_name, test_set, "inference.log")
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
                        print(test_set, seed, line)

                        utt_mse.append(float(line[-12]))
                        utt_lcc.append(float(line[-10]))
                        utt_srcc.append(float(line[-8]))
                        sys_mse.append(float(line[-5]))
                        sys_lcc.append(float(line[-3]))
                        sys_srcc.append(float(line[-1]))

                        break

        table.add_row(
            [
                test_set,
                f"{np.mean(np.array(utt_mse)):.3f}",
                f"{np.mean(np.array(utt_lcc)):.3f}",
                f"{np.mean(np.array(utt_srcc)):.3f}",
                f"{np.mean(np.array(sys_mse)):.3f}",
                f"{np.mean(np.array(sys_lcc)):.3f}",
                f"{np.mean(np.array(sys_srcc)):.3f}",
            ]
        )

        string.append(f"{np.mean(np.array(utt_mse)):.3f}")
        string.append(f"{np.mean(np.array(utt_lcc)):.3f}")
        string.append(f"{np.mean(np.array(utt_srcc)):.3f}")
        if test_set not in ["nisqa_FOR", "nisqa_LIVETALK", "nisqa_P501", "tmhintqi_test", "vmc23_track3_test"]:
            string.append(f"{np.mean(np.array(sys_mse)):.3f}")
            string.append(f"{np.mean(np.array(sys_lcc)):.3f}")
            string.append(f"{np.mean(np.array(sys_srcc)):.3f}")

    # print(table)

    print(" ".join(string))


if __name__ == "__main__":
    main()
