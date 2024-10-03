#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2024 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

"""Data preparation for multiple datasets."""

import argparse
import csv
import logging
import os
import sys

from sheet.utils import read_csv


def main():
    """Run training process."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--original-paths",
        nargs="+",
        required=True,
        type=str,
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
    metadata = []
    count = 0
    for original_path in args.original_paths:
        filelist, _ = read_csv(original_path, dict_reader=True)
        for line in filelist:
            if len(line) == 0:
                continue
            line["domain_idx"] = count
            line["system_id"] = f"{count}_{line['system_id']}"
            line["sample_id"] = f"{count}_{line['sample_id']}"
            metadata.append(line)
        count += 1

    # write csv
    logging.info("Writing output csv file.")
    fieldnames = ["wav_path", "score", "system_id", "sample_id", "domain_idx"]
    with open(args.out, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for line in metadata:
            writer.writerow(line)

if __name__ == "__main__":
    main()
