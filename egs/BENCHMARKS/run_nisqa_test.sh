#!/usr/bin/env bash

# Copyright 2024 Wen-Chin Huang (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# basic settings
stage=-1       # stage to start
stop_stage=100 # stage to stop
verbose=1      # verbosity level (lower is less info)
n_gpus=1       # number of gpus in training
seed=1337

conf=conf/ssl-mos-wav2vec2.yaml
meta_model_conf=conf/stacking_ridge.yaml

# dataset configuration
nisqa_db_root=/data/group1/z44476r/Corpora/nisqa/NISQA_Corpus

# training related setting
tag=""     # tag for directory to save model
           
# decoding related setting
test_sets="LIVETALK FOR P501"
checkpoint=""               # checkpoint path to be used for decoding
                            # if not provided, the latest one will be used
                            # (e.g. <path>/<to>/checkpoint-400000steps.pkl)
model_averaging="False"
use_stacking="False"
meta_model_checkpoint=""
                                       
# shellcheck disable=SC1091
. utils/parse_options.sh || exit 1;

set -euo pipefail

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    echo "stage -1: Data and Pretrained Model Download"

    local/data_download.sh ${db_root}
fi

mkdir -p "data"
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "stage 0: Data preparation"

    for test_set in ${test_sets}; do
        ../nisqa/local/data_prep.py \
            --original-path "${nisqa_db_root}/NISQA_TEST_${test_set}/NISQA_TEST_${test_set}_file.csv" --wavdir "${nisqa_db_root}/NISQA_TEST_${test_set}/deg" --out "data/nisqa_${test_set}.csv" 
    done
fi

if [ "${stage}" -le 1 ] && [ "${stop_stage}" -ge 1 ]; then
    echo "Stage 1: Inference"
    # shellcheck disable=SC2012

    if [ -z ${tag} ]; then
        expname="$(basename ${conf%.*})-${seed}"
    else
        expname="${tag}-${seed}"
    fi
    expdir=exp/${expname}

    if [ "${use_stacking}" = "True" ]; then
        [ -z "${meta_model_checkpoint}" ] && meta_model_checkpoint="${expdir}/meta_model.pkl"
        outdir="${expdir}/results/stacking-model"
    elif [ "${model_averaging}" = "True" ]; then
        outdir="${expdir}/results/model-averaging"
    else
        [ -z "${checkpoint}" ] && checkpoint="${expdir}/checkpoint-best.pkl"
        outdir="${expdir}/results/$(basename "${checkpoint}" .pkl)"
    fi

    for name in ${test_sets}; do
        name="nisqa_${name}"
        [ ! -e "${outdir}/${name}" ] && mkdir -p "${outdir}/${name}"
        [ "${n_gpus}" -gt 1 ] && n_gpus=1
        echo "Inference start. See the progress via ${outdir}/${name}/inference.log."
        ${cuda_cmd} --gpu "${n_gpus}" "${outdir}/${name}/inference.log" \
            inference.py \
                --config "${expdir}/config.yml" \
                --csv-path "data/${name}.csv" \
                --checkpoint "${checkpoint}" \
                --outdir "${outdir}/${name}" \
                --model-averaging "${model_averaging}" \
                --use-stacking "${use_stacking}" \
                --meta-model-checkpoint "${meta_model_checkpoint}" \
                --verbose "${verbose}"
        echo "Successfully finished inference of ${name} set."
        grep "UTT" "${outdir}/${name}/inference.log"
    done
    echo "Successfully finished inference."
fi