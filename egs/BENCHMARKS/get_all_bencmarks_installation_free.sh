#!/usr/bin/env bash

# Copyright 2024 Wen-Chin Huang (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

db_root=
datadir=data

stage=-1       # stage to start
stop_stage=0   # stage to stop

# shellcheck disable=SC1091
. utils/parse_options.sh || exit 1;

set -euo pipefail

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    echo "stage -1: Download data for all benchmark sets"

    ../bvcc/local/data_download.sh ${db_root}/bvcc
    ../bc19/local/data_download.sh ${db_root}/bc19
    ../somos/local/data_download.sh ${db_root}/somos
    ../singmos/local/data_download.sh ${db_root}/singmos
    ../nisqa/local/data_download.sh ${db_root}/nisqa
    ../tmhint_qi/local/data_download.sh ${db_root}/tmhint_qi
    ../vmc23/local/data_download.sh ${db_root}/vmc23

    echo "Please follow instructions in bvcc, bc19 to finish the download process."
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "stage 0: Data preparation for all benchmark sets"

    mkdir -p "${datadir}"

    # bvcc
    ../bvcc/local/data_prep.py \
        --original-path "${db_root}/bvcc/sets/TESTSET" --wavdir "${db_root}/wav" --out "${datadir}/bvcc_test.csv"

    # bc19
    ../bc19/local/data_prep.py \
        --original-path "${db_root}/bc19/sets/TESTSET" --wavdir "${db_root}/bc19/wav" --out "${datadir}/bc19_test.csv"

    # somos
    ../somos/local/data_prep.py \
        --original-path "${db_root}/somos/training_files/split1/clean/TESTSET" --wavdir "${db_root}/somos/audios" --out "${datadir}/somos_test.csv"

    # singmos
    ../singmos/local/data_prep.py \
        --original-path "${db_root}/singmos/sets/eval_mos_list.txt" --wavdir "${db_root}/wav" --out "${datadir}/singmos_test.csv"

    # nisqa
    for test_set in LIVETALK FOR P501; do
        ../nisqa/local/data_prep.py \
            --original-path "${db_root}/nisqa/NISQA_TEST_${test_set}/NISQA_TEST_${test_set}_file.csv" \
            --wavdir "${db_root}/nisqa/NISQA_TEST_${test_set}/deg" \
            --out "${datadir}/nisqa_${test_set}.csv"
    done

    # tmhint-qi
    ../tmhint-qi/local/data_prep.py \
        --original-path "${db_root}/tmhint_qi/raw_data.csv" --wavdir "${db_root}/test" --setname "test" --out "${datadir}/tmhintqi_test.csv"

    # vmc23
    for track in track1a track1b track2 track3; do
        if [ "${track}" = "track1a" ] || [ "${track}" = "track1b" ]; then
            _track=track1
        else
            _track="${track}"
        fi
        ../vmc23/local/data_prep.py \
            --original-path "${db_root}/vmc23/${_track}" \
            --wavdir "${db_root}/vmc23/${_track}" \
            --answer_path "../vmc23/answers/${_track}_answer.txt" \
            --track "${track}" \
            --out "${datadir}/vmc23_${track}_test.csv"
    done
fi
