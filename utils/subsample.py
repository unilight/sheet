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
        "--ratio",
        type=float,
        default=1.0,
        help=("ratio to sub-sample. if 1.0, then use whole dataset."),
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

    # subsample based on sample ids
    original_sample_ids = list(set([line["sample_id"] for line in filelist]))
    logging.info(f"Original unique sample size: {len(original_sample_ids)}")

    # randomly subsample based on num-total-samples
    if args.num_samples >= 0:
        n_samples = args.num_samples
    # randomly subsample based on ratio
    else:
        if args.ratio < 1.0:
            n_samples = int(len(original_sample_ids) * args.ratio)
    subsampled_sample_id = set(random.sample(original_sample_ids, n_samples))
    subsampled_filelist = [line for line in filelist if line["sample_id"] in subsampled_sample_id]
    logging.info(f"Subsampled size: {len(subsampled_filelist)}")

    # write csv
    logging.info("Writing output csv file.")
    with open(args.out, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for line in subsampled_filelist:
            writer.writerow(line)

if __name__ == "__main__":
    main()
