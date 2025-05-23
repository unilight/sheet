#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2024 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

"""Data preparation for VoiceMOS Challenge 2023 tracks."""

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
        "--track",
        required=True,
        type=str,
        help=("track. this is only used for track 1a/1b"),
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
    # Track 1: VoiceMOS2023Track1-A-AD_test_0026,4.42105263157895
    # Track 2: VoiceMOS2023Track2-B01-SF1-CDF1-30001,2.857142857142857
    # Track 3: VoiceMOS2023Track3-Noisy_snr5_white_TMHINT_g4_32_03,4.0
    #       or VoiceMOS2023Track3-clean_TMHINT_b1_01_03,5.0
    logging.info("Preparing metadata.")
    metadata = []
    for line in filelist:
        if len(line) == 0:
            continue
        sample_id = line[0]
        score = float(line[1])

        # decide system id based on track
        if "Track1" in sample_id:
            system_id = "-".join(sample_id.split("_")[0].split("-")[1:])  # "A-AD"
            if args.track == "track1a" and system_id.split("-")[-1] == "AD":
                continue
            if args.track == "track1b" and system_id.split("-")[-1] == "NEB":
                continue
        elif "Track2" in sample_id:
            system_id = sample_id.split("-")[1]  # "B01"
        elif "Track3" in sample_id:
            system_id = "_".join(
                sample_id.split("-")[1].split("_")[:-3]
            )  # "Noisy_snr5_white_TMHINT" or "clean_TMHINT"
            if system_id == "TMHINT":
                continue  # only accept "clean_TMHINT" but not "TMHINT"

        # get wav path
        wav_path = os.path.join(args.wavdir, sample_id + ".wav")

        # if resample and resample is necessary
        if (
            args.resample
            and librosa.get_samplerate(wav_path) != args.target_sampling_rate
        ):
            # check whether soundfile has been imported
            if "soundfile" not in sys.modules:
                import soundfile as sf

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

        metadata.append(item)

    # average score
    if args.avg_score_only:
        # take average score
        sample_scores = defaultdict(list)
        for item in metadata: # loop through metadata
            sample_scores[item["sample_id"]].append(float(item["score"]))
        sample_avg_score = {
            sample_id: np.mean(np.array(scores))
            for sample_id, scores in sample_scores.items()
        } # take average
        for i, item in enumerate(metadata): # fill back into metadata
            metadata[i]["avg_score"] = sample_avg_score[item["sample_id"]]
        
        new_metadata = {}  # {sample_id: item}
        for item in metadata:
            sample_id = item["sample_id"]
            if not sample_id in new_metadata:
                new_metadata[sample_id] = {
                    k: v
                    for k, v in item.items()
                    if k not in ["listener_id", "listener_idx", "score"]
                }

        metadata = list(new_metadata.values())

    # write csv
    logging.info("Writing output csv file.")
    fieldnames = ["wav_path", "system_id", "sample_id"]
    if args.avg_score_only:
        fieldnames.append("avg_score")
    else:
        fieldnames.append("score")
    with open(args.out, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for line in metadata:
            writer.writerow(line)


if __name__ == "__main__":
    main()
