#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2024 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

"""Data preparation for TMHINT-QI."""

import argparse
import csv
import logging
import os
import random
import sys

from sheet.utils import read_csv, find_files


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
        "--wavdir",
        required=True,
        type=str,
        help=(
            "directory of the waveform files. This is needed because wav paths in BVCC metadata files do not contain the wav directory."
        ),
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
        "--out",
        required=True,
        type=str,
        help=("output csv file path."),
    )
    parser.add_argument(
        "--seed",
        default=1337,
        type=int,
        help=("Random seed. This is used to get consistent random sampling results."),
    )
    parser.add_argument(
        "--domain-idx",
        type=int,
        default=None,
        help=("domain ID.")
    )
    args = parser.parse_args()

    # set logger
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )

    random.seed(args.seed)

    # read csv
    logging.info("Reading original csv file.")
    filelist, _ = read_csv(args.original_path, dict_reader=True)

    # Get the wav files first
    wav_files = find_files(args.wavdir)  # this returns the full path

    # shuffle and split
    random.shuffle(wav_files)
    dev_num = int(len(wav_files) * args.dev_ratio)
    if args.setname == "train":
        wav_files = wav_files[dev_num:]
    elif args.setname == "dev":
        wav_files = wav_files[:dev_num]

    # scan through filelist and only use those in the wav_files list
    # each line looks like this:
    # idx,method,uttr,snr,noise,quality_score,intelligibility_score,file_name
    logging.info("Preparing metadata.")
    metadata = []
    for line in filelist:
        if len(line) == 0:
            continue
        method = line["method"]
        snr = line["snr"]
        noise = line["noise"]
        system_id = f"{method}-SNR{snr}-{noise}"
        sample_id = line["file_name"]
        wav_path = os.path.join(args.wavdir, sample_id + ".wav")
        if not wav_path in wav_files or not os.path.isfile(wav_path):
            continue
        if len(line["quality_score"]) == 0:
            continue
        score = int(line["quality_score"])
        item = {
            "wav_path": wav_path,
            "score": score,
            "system_id": system_id,
            "sample_id": sample_id,
        }
        # append domain ID if given
        if args.domain_idx is not None:
            item["domain_idx"] = args.domain_idx
        metadata.append(item)
    metadata.sort(key=lambda x: x["wav_path"])

    # write csv
    logging.info("Writing output csv file.")
    fieldnames = ["wav_path", "score", "system_id", "sample_id"]
    if args.domain_idx is not None:
        fieldnames.append("domain_idx")
    with open(args.out, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for line in metadata:
            writer.writerow(line)


if __name__ == "__main__":
    main()
