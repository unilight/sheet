#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2024 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

"""Data preparation for BC19."""

import argparse
from collections import defaultdict
import csv
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
        wav_path = line[1] + ".wav"
        sample_id = os.path.splitext(wav_path.split("-")[1])[0]
        score = int(line[2])
        listener_id = line[4]

        wav_path = os.path.join(args.wavdir, wav_path)

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
            "listener_id": listener_id,
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
    fieldnames = [
        "wav_path",
        "system_id",
        "sample_id",
    ]
    if args.avg_score_only:
        fieldnames.append("avg_score")
    else:
        fieldnames.append("score")
        fieldnames.append("listener_id")
    with open(args.out, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for line in metadata:
            writer.writerow(line)


if __name__ == "__main__":
    main()
