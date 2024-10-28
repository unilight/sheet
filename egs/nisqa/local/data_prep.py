#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2024 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

"""Data preparation for NISQA."""

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
            "directory of the waveform files. This is needed because wav paths in the metadata files, the wav dir is not contained."
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
        "--domain-idx",
        type=int,
        default=None,
        help=("domain ID.")
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
    filelist, _ = read_csv(args.original_path, dict_reader=True)

    # prepare. header:
    # db,con,file,con_description,filename_deg,filename_ref,source,lang,votes,mos,noi,col,dis,loud,noi_std,col_std,dis_std,loud_std,mos_std,filepath_deg,filepath_ref,filter,timeclipping,wbgn,p50mnru,bgn,clipping,arb_filter,asl_in,asl_out,codec1,codec2,codec3,plcMode1,plcMode2,plcMode3,wbgn_snr,bgn_snr,tc_fer,tc_nburst,cl_th,bp_low,bp_high,p50_q,bMode1,bMode2,bMode3,FER1,FER2,FER3,asl_in_level,asl_out_level
    logging.info("Preparing metadata.")
    metadata = []
    for line in filelist:
        if len(line) == 0:
            continue
        system_id = line["con"]
        wav_path = line["filename_deg"]
        complete_wav_path = os.path.join(args.wavdir, wav_path)
        sample_id = os.path.splitext(wav_path)[0]
        score = float(line["mos"])

        # if resample and resample is necessary
        if (
            args.resample
            and librosa.get_samplerate(complete_wav_path) != args.target_sampling_rate
        ):
            # check whether soundfile has been imported
            if "soundfile" not in sys.modules:
                import soundfile as sf

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
        # append domain ID if given
        if args.domain_idx is not None:
            item["domain_idx"] = args.domain_idx
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
        fieldnames.append("score", "listener_id")
    if args.domain_idx is not None:
        fieldnames.append("domain_idx")
    with open(args.out, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for line in metadata:
            writer.writerow(line)


if __name__ == "__main__":
    main()
