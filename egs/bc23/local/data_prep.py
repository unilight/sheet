#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2025 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

"""Data preparation for BC 2023."""

import argparse
from collections import defaultdict
import csv
from distutils.util import strtobool
import logging
import os
import sys
from tqdm import tqdm

import numpy as np
import soundfile as sf

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
        "--out_a",
        required=True,
        type=str,
        help=("output csv file path for track a."),
    )
    parser.add_argument(
        "--out_b",
        required=True,
        type=str,
        help=("output csv file path for track b."),
    )
    parser.add_argument(
        "--resample",
        action="store_true",
        help=("whether to perform resampling or not."),
    )
    parser.add_argument(
        "--target-sampling-rate",
        type=int,
        help=("target sampling rate."),
    )
    parser.add_argument(
        "--resample-backend",
        type=str,
        default="librosa",
        choices=["librosa"],
        help=("resample backend."),
    )
    parser.add_argument(
        "--target-wavdir",
        type=str,
        help=("directory of the resampled waveform files."),
    )
    parser.add_argument(
        "--avg-score-only",
        action="store_true",
        help=("generate average score only. set for test set preparation.")
    )
    args = parser.parse_args()

    # set logger
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )

    # make resampled dir and dynamic import
    if args.resample:
        os.makedirs(args.target_wavdir, exist_ok=True)
        if args.resample_backend == "librosa":
            import librosa

    # read answer csv
    logging.info("Reading answer csv file.")
    filelist, _ = read_csv(args.answer_path)

    # prepare. each line looks like this:
    # VoiceMOS2023Track1-A-AD_test_0026,4.42105263157895
    logging.info("Preparing metadata.")
    metadata_a = []
    metadata_b = []
    for line in tqdm(filelist):
        if len(line) == 0:
            continue
        sample_id = line[0]
        score = float(line[1])
        system_id = "-".join(sample_id.split("_")[0].split("-")[1:])  # "A-AD"

        # get wav path
        wav_path = os.path.join(args.wavdir, sample_id + ".wav")

        # if resample and resample is necessary
        if (
            args.resample
            and librosa.get_samplerate(wav_path) != args.target_sampling_rate
        ):
            resampled_wav_path = os.path.join(args.target_wavdir, sample_id + ".wav")
            # resample and write if not exist yet
            if not os.path.isfile(resampled_wav_path):
                if args.resample_backend == "librosa":
                    resampled_wav, _ = librosa.load(
                        wav_path, sr=args.target_sampling_rate
                    )
                sf.write(
                    resampled_wav_path,
                    resampled_wav,
                    samplerate=args.target_sampling_rate,
                )
            wav_path = resampled_wav_path

        item = {
            "wav_path": wav_path,
            "score": score,
            "system_id": system_id,
            "sample_id": sample_id,
        }

        # track A -> NEB
        if system_id.split("-")[-1] == "NEB":
            metadata_a.append(item)
        # track B -> AD
        if system_id.split("-")[-1] == "AD":
            metadata_b.append(item)
        
    # write csv
    logging.info("Writing output csv file.")
    fieldnames = ["wav_path", "system_id", "sample_id", "score"]
    for outpath, _metadata in [[args.out_a, metadata_a], [args.out_b, metadata_b]]:
        with open(outpath, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for line in _metadata:
                writer.writerow(line)


if __name__ == "__main__":
    main()
