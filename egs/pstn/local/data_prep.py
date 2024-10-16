#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2024 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

"""Data preparation for PSTN."""

import argparse
import csv
import logging
import os
import random
import soundfile as sf
import sys
from tqdm import tqdm

from sheet.utils import read_csv


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
        "--num-total-samples",
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

    # set logger
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )

    # set seed
    random.seed(args.seed)

    # make resampled dir and dynamic import
    if args.resample:
        os.makedirs(args.target_wavdir, exist_ok=True)
        if args.resample_backend == "librosa":
            import librosa

    # read csv
    logging.info("Reading original csv file.")
    filelist, _ = read_csv(args.original_path, dict_reader=True)

    # randomly subsample based on num-total-samples
    if args.num_total_samples >= 0:
        filelist = random.sample(filelist, args.num_total_samples)

    # prepare. each line looks like this:
    # filename,MOS,std,95%CI,votes
    logging.info("Preparing metadata.")
    metadata = []
    listener_idxs, count = {}, 0
    for line in tqdm(filelist):
        if len(line) == 0:
            continue
        wav_path = line["filename"]
        complete_wav_path = os.path.join(args.wavdir, line["filename"])
        sample_id = os.path.splitext(line["filename"])[0].split("_")[0]
        score = float(line["MOS"])
        system_id = sample_id  # there is no system ID information in PSTN...

        # if resample and resample is necessary
        if (
            args.resample
            and librosa.get_samplerate(complete_wav_path) != args.target_sampling_rate
        ):
            resampled_wav_path = os.path.join(args.target_wavdir, wav_path)
            # resample and write if not exist yet
            if not os.path.isfile(resampled_wav_path):
                if args.resample_backend == "librosa":
                    resampled_wav, _ = librosa.load(
                        complete_wav_path, sr=args.target_sampling_rate
                    )
                sf.write(
                    resampled_wav_path,
                    resampled_wav,
                    samplerate=args.target_sampling_rate,
                )
            complete_wav_path = resampled_wav_path

        item = {
            "wav_path": complete_wav_path,
            "score": score,
            "system_id": system_id,
            "sample_id": sample_id,
        }
        metadata.append(item)

    # shuffle and split
    random.shuffle(metadata)
    dev_num = int(len(metadata) * args.dev_ratio)
    if args.setname == "train":
        metadata = metadata[dev_num:]
    elif args.setname == "dev":
        metadata = metadata[:dev_num]

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
