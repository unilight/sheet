#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2024 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

"""Data preparation for the Tencent corpus."""

import argparse
import csv
import logging
import os
import random
import sys

from sheet.utils import read_csv


def main():
    """Run training process."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--original-path",
        required=True,
        nargs="+",
        help=("original csv file paths. For the Tencent corpus we take two."),
    )
    parser.add_argument(
        "--wavdir",
        required=True,
        type=str,
        help=(
            "directory of the waveform files. This is needed because wav paths in BVCC metadata files do not contain the wav directory."
        ),
    )
    parser.add_argument(
        "--out",
        required=True,
        type=str,
        help=("output csv file path."),
    )
    parser.add_argument(
        "--setname",
        required=True,
        type=str,
        choices=["train", "dev", "test"],
        help=(
            "setname. Since there is no dev set, we need to randomly sample dev set on our own."
        ),
    )
    parser.add_argument(
        "--dev_ratio",
        default=0.1,
        type=float,
        help=("The ratio of the dev set. Default: 0.1"),
    )
    parser.add_argument(
        "--seed",
        default=1337,
        type=int,
        help=("Random seed. This is used to get consistent random sampling results."),
    )
    args = parser.parse_args()

    # set logger
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )

    # read csv
    metadata = []
    for original_path in args.original_path:
        logging.info(f"Reading original csv file {original_path}")
        filelist, _ = read_csv(original_path, dict_reader=True)

        # prepare. each line looks like this:
        # deg_wav,mos
        current_metadata = []
        for line in filelist:
            if len(line) == 0:
                continue
            wav_path = line["deg_wav"].replace("./", "")
            score = float(line["mos"])
            sample_id = wav_path.replace(".wav", "").replace(os.sep, "_")
            system_id = sample_id  # no system ID information
            item = {
                "wav_path": os.path.join(args.wavdir, wav_path),
                "score": score,
                "system_id": system_id,
                "sample_id": sample_id,
            }
            current_metadata.append(item)

        # shuffle and split
        random.shuffle(current_metadata)
        dev_num = int(len(current_metadata) * args.dev_ratio)
        if args.setname == "train":
            current_metadata = current_metadata[dev_num:]
        elif args.setname == "dev":
            current_metadata = current_metadata[:dev_num]

        metadata.extend(current_metadata)
    metadata.sort(key=lambda x: x["wav_path"])

    # write csv
    logging.info("Writing output csv file.")
    fieldnames = ["wav_path", "score", "system_id", "sample_id"]
    with open(args.out, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for line in metadata:
            writer.writerow(line)


if __name__ == "__main__":
    main()
