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
vmc23_db_root=/data/group1/z44476r/Corpora/vmc23    # change this to your dataset folder
# vmc23_db_root=../vmc23/downloads
datadir="../vmc23/data"
target_sampling_rate=16000

# training related setting
tag=""     # tag for directory to save model

datastore_path=

# decoding related setting
test_sets="vmc23_track1a_test vmc23_track1b_test vmc23_track2_test vmc23_track3_test"
checkpoint=""               # checkpoint path to be used for decoding
                            # if not provided, the latest one will be used
                            # (e.g. <path>/<to>/checkpoint-400000steps.pkl)
model_averaging="False"
use_stacking="False"
meta_model_checkpoint=""
np_inference_mode=""
                                       
# shellcheck disable=SC1091
. utils/parse_options.sh || exit 1;

set -euo pipefail

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    echo "stage -1: Data and Pretrained Model Download"

    ../vmc23/local/data_download.sh ${vmc23_db_root}
fi


mkdir -p "${datadir}"
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "stage 0: Data preparation"

    for track in track1a track1b track2 track3; do
        if [ "${track}" = "track1a" ] || [ "${track}" = "track1b" ]; then
            _track=track1
        else
            _track="${track}"
        fi
        ../vmc23/local/data_prep.py \
            --original-path "${vmc23_db_root}/${_track}" \
            --wavdir "${vmc23_db_root}/${_track}" \
            --answer_path "../vmc23/answers/${_track}_answer.txt" \
            --track "${track}" \
            --out "${datadir}/vmc23_${track}_test.csv" \
            --resample --target-sampling-rate "${target_sampling_rate}" --target-wavdir "${vmc23_db_root}/${_track}_${target_sampling_rate}"
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

########## EXPERIMENTAL FEATURE. DEPRECATED. #########################

if [ "${stage}" -le 2 ] && [ "${stop_stage}" -ge 2 ]; then
    echo "Stage 2: All domain idxs inference"
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
        outdir="${expdir}/results/all_domain_idxs_$(basename "${checkpoint}" .pkl)"
    fi

    for name in ${test_sets}; do
        [ ! -e "${outdir}/${name}" ] && mkdir -p "${outdir}/${name}"
        [ "${n_gpus}" -gt 1 ] && n_gpus=1
        echo "Inference start. See the progress via ${outdir}/${name}/inference.log."
        ${cuda_cmd} --gpu "${n_gpus}" "${outdir}/${name}/inference.log" \
            inference_loop_all_domain_idxs.py \
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

######################################################################

if [ "${stage}" -le 3 ] && [ "${stop_stage}" -ge 3 ]; then
    echo "Stage 3: Non-parametric inference"
    # shellcheck disable=SC2012

    if [ -z ${tag} ]; then
        expname="$(basename ${conf%.*})-${seed}"
    else
        expname="${tag}-${seed}"
    fi
    expdir=exp/${expname}

    [ -z "${checkpoint}" ] && checkpoint="${expdir}/checkpoint-best.pkl"
    outdir="${expdir}/results/np_$(basename "${checkpoint}" .pkl)/${np_inference_mode}"

    if [ -z ${datastore_path} ]; then
        datastore_path="${expdir}/datastore/$(basename "${checkpoint}" .pkl)/datastore.h5"
    fi

    for name in ${test_sets}; do
        [ ! -e "${outdir}/${name}" ] && mkdir -p "${outdir}/${name}"
        [ "${n_gpus}" -gt 1 ] && n_gpus=1
        echo "Inference start. See the progress via ${outdir}/${name}/inference.log."
        ${cuda_cmd} --gpu "${n_gpus}" "${outdir}/${name}/inference.log" \
            nonparametric_inference.py \
                --config "${expdir}/config.yml" \
                --datastore "${datastore_path}" \
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