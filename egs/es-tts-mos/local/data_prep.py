#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2025 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

"""
Data preparation for ES-TTS-MOS, described in the paper titled
`A Dataset for Automatic Assessment of TTS Quality in Spanish`
by Alejandro Sosa Welford, Leonardo Pepino.
https://arxiv.org/abs/2507.01805
"""

import argparse
from collections import defaultdict
import csv
from distutils.util import strtobool
import logging
import os
import sys
from tqdm import tqdm

import numpy as np


# The following function(s) is(are) the same as in sheet.utils.utils
# copied here for installation-free data preparation
def read_csv(path, dict_reader=False, lazy=False):
    with open(path, newline="") as csvfile:
        if dict_reader:
            reader = csv.DictReader(csvfile)
            fieldnames = reader.fieldnames
        else:
            reader = csv.reader(csvfile)
            fieldnames = None

        if lazy:
            contents = reader
        else:
            contents = [line for line in reader]

    return contents, fieldnames

def str2bool(value: str) -> bool:
    return bool(strtobool(value))


def main():
    """Run data preprocessing."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--original-path",
        required=True,
        type=str,
        help=("original csv file path."),
    )
    parser.add_argument(
        "--wavdir",
        required=True,
        type=str,
        help=("directory of the waveform files."),
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
    logging.info("Reading original csv file.")
    filelist, _ = read_csv(args.original_path, dict_reader=True)

    metadata = []
    for line in tqdm(filelist):
        sample_id = line["sample_id"]
        system_id = line["system_id"]
        score = float(line["score"])
        speaker_id = line["speaker_id"]
        duration_ms = line["duration_ms"]
        wav_path = os.path.join(args.wavdir, line["wav_path"])

        item = {
            "wav_path": wav_path,
            "score": score,
            "system_id": system_id,
            "sample_id": sample_id,
            "speaker_id": speaker_id,
            "duration_ms": duration_ms,
        }

        metadata.append(item)

    # write csv
    logging.info("Writing output csv file.")
    fieldnames = [
        "wav_path",
        "system_id",
        "sample_id",
        "speaker_id",
        "duration_ms",
        "score",
    ]
    with open(args.out, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for line in metadata:
            writer.writerow(line)


if __name__ == "__main__":
    main()
