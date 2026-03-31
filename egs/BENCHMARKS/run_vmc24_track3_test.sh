#!/usr/bin/env bash

# Copyright 2025 Wen-Chin Huang (Nagoya University)
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
vmc24_track3_db_root=/data/group1/z44476r/Corpora/vmc24_track3 # change this to your dataset folder
datadir="../vmc24_track3/data"
domain_idx=0
target_sampling_rate=16000

# training related setting
exp_root="exp" # Default. Will be dynamically overwritten if --checkpoint is provided.
tag=""         # tag for directory to save model
           
# decoding related setting
test_sets="vmc24_track3_test"
checkpoint=""               # checkpoint path to be used for decoding
                            # if not provided, the latest one will be used
                            # (e.g. <path>/<to>/checkpoint-400000steps.pkl)
model_averaging="False"
use_stacking="False"
meta_model_checkpoint=""
np_inference_mode=
                                       
# shellcheck disable=SC1091
. utils/parse_options.sh || exit 1;

set -euo pipefail

# Infer expdir and exp_root based on the checkpoint, or build it from conf/tag
if [ -n "${checkpoint}" ]; then
    expdir="$(dirname "${checkpoint}")"
    exp_root="$(dirname "${expdir}")"
else
    if [ -z "${tag}" ]; then
        expname="$(basename "${conf%.*}")-${seed}"
    else
        expname="${tag}-${seed}"
    fi
    expdir="${exp_root}/${expname}"
    checkpoint="${expdir}/checkpoint-best.pkl"
fi

if [ "${stage}" -le -1 ] && [ "${stop_stage}" -ge -1 ]; then
    echo "stage -1: Data and Pretrained Model Download"

    ../vmc24_track3/local/data_download.sh "${vmc24_track3_db_root}"
fi


mkdir -p "${datadir}"
if [ "${stage}" -le 0 ] && [ "${stop_stage}" -ge 0 ]; then
    echo "stage 0: Data preparation"

    ../vmc24_track3/local/data_prep.py \
        --wavdir "${vmc24_track3_db_root}/DATA/wav" \
        --answer_path "../vmc24_track3/answers/track3_answer.txt" \
        --out "${datadir}/vmc24_track3_test.csv"
fi

if [ "${stage}" -le 1 ] && [ "${stop_stage}" -ge 1 ]; then
    echo "Stage 1: Inference"

    if [ "${use_stacking}" = "True" ]; then
        [ -z "${meta_model_checkpoint}" ] && meta_model_checkpoint="${expdir}/meta_model.pkl"
        outdir="${expdir}/results/stacking-model"
    elif [ "${model_averaging}" = "True" ]; then
        outdir="${expdir}/results/model-averaging"
    else
        outdir="${expdir}/results/$(basename "${checkpoint}" .pkl)"
    fi

    for name in ${test_sets}; do
        [ ! -e "${outdir}/${name}" ] && mkdir -p "${outdir}/${name}"
        [ "${n_gpus}" -gt 1 ] && n_gpus=1
        echo "Inference start. See the progress via ${outdir}/${name}/inference.log."
        ${cuda_cmd} --gpu "${n_gpus}" "${outdir}/${name}/inference.log" \
            inference.py \
                --config "${expdir}/config.yml" \
                --csv-path "${datadir}/${name}.csv" \
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

if [ "${stage}" -le 3 ] && [ "${stop_stage}" -ge 3 ]; then
    echo "Stage 3: Non-parametric inference"

    outdir="${expdir}/results/np_$(basename "${checkpoint}" .pkl)/${np_inference_mode}"

    for name in ${test_sets}; do
        [ ! -e "${outdir}/${name}" ] && mkdir -p "${outdir}/${name}"
        [ "${n_gpus}" -gt 1 ] && n_gpus=1
        echo "Inference start. See the progress via ${outdir}/${name}/inference.log."
        ${cuda_cmd} --gpu "${n_gpus}" "${outdir}/${name}/inference.log" \
            nonparametric_inference.py \
                --config "${expdir}/config.yml" \
                --datastore "${expdir}/datastore/$(basename "${checkpoint}" .pkl)/datastore.h5" \
                --csv-path "${datadir}/${name}.csv" \
                --checkpoint "${checkpoint}" \
                --outdir "${outdir}/${name}" \
                --np-inference-mode "${np_inference_mode}" \
                --verbose "${verbose}"
        echo "Successfully finished inference of ${name} set."
        grep "UTT" "${outdir}/${name}/inference.log"
    done
    echo "Successfully finished inference."
fi