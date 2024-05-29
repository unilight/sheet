#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2024 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

"""Combine multiple datasets (represented with csv files) into one."""

import argparse
import csv
import librosa
import logging
import os
import soundfile as sf
import sys
from tqdm import tqdm

from sheet.utils import read_csv

def main():
    """Run training process."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--original-paths",
        required=True,
        type=str,
        nargs='+',
        help=("original csv file paths."),
    )
    parser.add_argument(
        "--out",
        required=True,
        type=str,
        help=("output csv file path."),
    )
    args = parser.parse_args()

    # set logger
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )

    # read csv
    logging.info("Reading original csv files.")
    originals = [
        read_csv(original_path, dict_reader=True)[0]
        for original_path in args.original_paths
    ]

    # take the union of all headers
    all_keys = set()
    for original in originals:
        for k in original[0].keys():
            all_keys.add(k)
    fieldnames = list(all_keys)

    # write csv
    logging.info("Writing output csv file.")
    with open(args.out, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for original in originals:
            for line in original:
                writer.writerow(line)

if __name__ == "__main__":
    main()
