#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2025 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

"""
Print results within one checkpoint with prettytable.

2025.9.1
This is for TASLP resubmission.
"""

import argparse
import csv
import sys
import os
from prettytable import PrettyTable, MARKDOWN

synthetic_sets = [
    "bvcc_test",
    "somos_test",
    "bc19_test",
    "bc23a_test",
    "bc23b_test",
    "svcc23_test",
    "singmos_test",
    "brspeechmos_test",
    "hablamos_test",
    "ttsds2_test",
]

distorted_sets = [
    "nisqa_FOR",
    "nisqa_LIVETALK",
    "nisqa_P501",
    "tmhintqi_test",
    "tmhint_qi_s_test",
    "tcd_voip_test",
    "vmc24_track3_test"
]

order = [
    "bvcc_test",
    "somos_test",
    "bc19_test",
    "bc23a_test",
    "bc23b_test",
    "svcc23_test",
    "singmos_test",
    "brspeechmos_test",
    "hablamos_test",
    "ttsds2_test",
    "nisqa_FOR",
    "nisqa_LIVETALK",
    "nisqa_P501",
    "tmhintqi_test",
    "tmhint_qi_s_test",
    "tcd_voip_test",
    "vmc24_track3_test"
]

def main():
    """Run training process."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dir",
        type=str,
    )
    parser.add_argument(
        "--type",
        type=str,
        default="srcc",
        choices=["srcc", "mse"],
    )
    parser.add_argument(
        "--all_utt",
        action="store_true"
    )
    args = parser.parse_args()

    ccs = {}
    logvars = {}

    for d in sorted(os.listdir(args.dir)):
        log_filepath = os.path.join(args.dir, d, "inference.log")
        if os.path.isfile(log_filepath):
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
                        if args.type == "srcc":
                            if d in synthetic_sets:
                                if args.all_utt:
                                    ccs[d] = float(line[-8])
                                else:
                                    ccs[d] = float(line[-1])
                            elif d in distorted_sets:
                                # ccs[d] = float(line[-10])
                                ccs[d] = float(line[-8])
                        elif args.type == "mse":
                            if d in synthetic_sets:
                                if args.all_utt:
                                    ccs[d] = float(line[-5])
                                else:
                                    ccs[d] = float(line[-12])
                            elif d in distorted_sets:
                                ccs[d] = float(line[-12])

                    if "Mean log variance" in line:
                        # 2025-06-24 14:55:52,572 (inference:364) INFO: Mean log variance: -1.084
                        logvars[d] = float(line.split(": ")[-1])

    print(" ".join(f"{ccs[d]:.3f}" for d in order))
    if len(logvars) != 0:
        print(" ".join(f"{logvars[d]:.3f}" for d in order))

if __name__ == "__main__":
    main()
