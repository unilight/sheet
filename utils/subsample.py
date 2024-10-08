#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2024 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

"""Subsampling a csv file"""

import argparse
import csv
import logging
import sys
import random

from sheet.utils import read_csv


def main():
    """Run training process."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--original-path",
        required=True,
        type=str,
        help=("original csv file path."),
    )
    parser.add_argument(
        "--out",
        required=True,
        type=str,
        help=("output csv file path."),
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=-1,
        help=("num of total samples to sub-sample. if <=0, then use whole dataset."),
    )
    parser.add_argument(
        "--seed",
        default=1337,
        type=int,
        help=("Random seed. This is used to get consistent random sampling results."),
    )
    args = parser.parse_args()

    # set seed
    random.seed(args.seed)

    # set logger
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )

    # read csv
    logging.info("Reading original csv file.")
    filelist, _ = read_csv(args.original_path, dict_reader=True)
    fieldnames = list(filelist[0].keys())

    # randomly subsample based on num-total-samples
    if args.num_samples >= 0:
        filelist = random.sample(filelist, args.num_samples)

    # write csv
    logging.info("Writing output csv file.")
    with open(args.out, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for line in filelist:
            writer.writerow(line)

if __name__ == "__main__":
    main()
