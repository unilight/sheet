#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2024 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

"""Data preparation for BVCC."""

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
        "--out",
        required=True,
        type=str,
        help=("output csv file path."),
    )
    parser.add_argument("--generate-listener-id", action="store_true")
    args = parser.parse_args()

    # set logger
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )

    # read csv
    logging.info("Reading original csv file.")
    filelist, _ = read_csv(args.original_path)

    # prepare. each line looks like this:
    # sys64e2f,sys64e2f-utt9c183cd.wav,4,VDP1ovyrBzg8_1,{}_30-39_bZPQE7w4Zl3g_Female_Valid_1_No
    logging.info("Preparing metadata.")
    metadata = []
    listener_idxs, count = {}, 0
    for line in filelist:
        if len(line) == 0:
            continue
        system_id = line[0]
        wav_path = line[1]
        sample_id = os.path.splitext(wav_path.split("-")[1])[0]
        score = int(line[2])
        listener_id = line[4]
        item = {
            "wav_path": os.path.join(args.wavdir, wav_path),
            "score": score,
            "system_id": system_id,
            "sample_id": sample_id,
            "listener_id": listener_id,
        }
        if args.generate_listener_id:
            if not listener_id in listener_idxs:
                listener_idxs[listener_id] = count
                count += 1
            item["listener_idx"] = listener_idxs[listener_id]
        metadata.append(item)

    # write csv
    logging.info("Writing output csv file.")
    fieldnames = ["wav_path", "score", "system_id", "sample_id", "listener_id"]
    if args.generate_listener_id:
        fieldnames.append("listener_idx")
    with open(args.out, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for line in metadata:
            writer.writerow(line)


if __name__ == "__main__":
    main()
