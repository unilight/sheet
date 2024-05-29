#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2024 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

"""Data preparation for SOMOS."""

import argparse
import csv
import librosa
import logging
import os
import soundfile as sf
import sys
from tqdm import tqdm

from sheet.utils import read_csv

def main():
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
        help=("directory of the waveform files. This is needed because wav paths in the metadata files do not contain the wav directory."),
    )
    parser.add_argument(
        "--out",
        required=True,
        type=str,
        help=("output csv file path."),
    )
    parser.add_argument(
        "--generate-listener-id",
        action='store_true'
    )
    parser.add_argument(
        "--resample",
        action='store_true',
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
    filelist, _ = read_csv(args.original_path, dict_reader=True)

    # prepare. each line looks like this:
    # systemId,utteranceId,choice,listenerId
    # 015,novel_2007_0098_015.wav,4,KEXM49572020611127
    logging.info("Preparing metadata.")
    metadata = []
    listener_idxs, count = {}, 0
    for line in tqdm(filelist):
        if len(line) == 0: continue
        system_id = line["systemId"]
        sample_id = line["utteranceId"]
        wav_path = os.path.join(args.wavdir, sample_id)
        score = int(line["choice"])
        listener_id = line["listenerId"]

        # if resample and resample is necessary
        if args.resample and librosa.get_samplerate(wav_path) != args.target_sampling_rate:
            resampled_wav_path = os.path.join(args.target_wavdir, sample_id)
            # resample and write if not exist yet
            if not os.path.isfile(resampled_wav_path):
                if args.resample_backend == "librosa":
                    resampled_wav, _ = librosa.load(wav_path, sr=args.target_sampling_rate)
                sf.write(resampled_wav_path, resampled_wav, samplerate=args.target_sampling_rate)
            wav_path = resampled_wav_path



        item = {
            "wav_path": wav_path,
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
    with open(args.out, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for line in metadata:
            writer.writerow(line)

if __name__ == "__main__":
    main()
