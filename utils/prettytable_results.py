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


def main():
    """Run training process."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir",
        required=True,
        type=str,
        help=("original csv file paths."),
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

                        table.add_row(
                            [
                                d,
                                float(line[-12]),
                                float(line[-10]),
                                float(line[-8]),
                                float(line[-5]),
                                float(line[-3]),
                                float(line[-1]),
                            ]
                        )
                        break

    print(table)


if __name__ == "__main__":
    main()
