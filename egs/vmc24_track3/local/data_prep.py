#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2025 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

"""
Data preparation for VoiceMOS Challenge 2024 track 3.
In MOS-Bench, we only consider the "OVRL" score.
"""

import argparse
from collections import defaultdict
import csv
from distutils.util import strtobool
import logging
import os
import sys

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
        "--wavdir",
        required=True,
        type=str,
        help=(
            "directory of the waveform files. This is needed because we put absolute paths in the csv files."
        ),
    )
    parser.add_argument(
        "--answer_path",
        required=True,
        type=str,
        help=("answer csv file path."),
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

    # read answer csv
    logging.info("Reading answer csv file.")
    data, _ = read_csv(args.answer_path, dict_reader=True)

    # format: systems,SIG,BAK,OVR
    logging.info("Preparing data.")
    metadata = []
    for line in data:
        if len(line) == 0:
            continue
        sample_id = line["systems"]
        system_id = sample_id
        score = float(line["OVR"])
        wav_path = os.path.join(
            args.wavdir, "voicemos2024-track3-" + sample_id + ".wav"
        )
        assert os.path.isfile(wav_path), f"{wav_path} does not exist."

        item = {
            "wav_path": wav_path,
            "score": score,
            "system_id": system_id,
            "sample_id": sample_id,
        }

        metadata.append(item)

    # write csv
    logging.info("Writing output csv file.")
    fieldnames = ["wav_path", "system_id", "sample_id", "score"]
    with open(args.out, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for line in metadata:
            writer.writerow(line)


if __name__ == "__main__":
    main()
