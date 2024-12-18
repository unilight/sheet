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
n_jobs=16      # number of parallel jobs in feature extraction
seed=1337

conf=conf/ssl-mos-wav2vec2.yaml

# dataset configuration
db_root=/data/group1/z44476r/Corpora/pstn/pstn_train  # change this to your dataset folder
target_sampling_rate=16000

# training related setting
tag=""     # tag for directory to save model
resume=""  # checkpoint path to resume training
           # (e.g. <path>/<to>/checkpoint-10000steps.pkl)
           
# decoding related setting
checkpoint=""               # checkpoint path to be used for decoding
                            # if not provided, the latest one will be used
                            # (e.g. <path>/<to>/checkpoint-400000steps.pkl)
model_averaging="False"
use_stacking="False"
meta_model_checkpoint=""
                                       
# shellcheck disable=SC1091
. utils/parse_options.sh || exit 1;

set -euo pipefail

train_set="pstn_train"
dev_set="pstn_dev"
test_sets="${dev_set}"

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    echo "stage -1: Data and Pretrained Model Download"

    # Just provide instructions?

fi

mkdir -p "data"
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "stage 0: Data preparation"

    mkdir -p "${db_root}/wav_${target_sampling_rate}" 

    # parse original csv file to an unified format
    local/data_prep.py \
        --original-path "${db_root}/pstn_train.csv" --wavdir "${db_root}" --setname "train" --out "data/${train_set}.csv" --seed "${seed}" \
        --resample --target-sampling-rate "${target_sampling_rate}" --target-wavdir "${db_root}/wav_${target_sampling_rate}"
    local/data_prep.py \
        --original-path "${db_root}/pstn_train.csv" --wavdir "${db_root}" --setname "dev" --out "data/${dev_set}.csv" --seed "${seed}" \
        --resample --target-sampling-rate "${target_sampling_rate}" --target-wavdir "${db_root}/wav_${target_sampling_rate}"
        
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "stage 1: Pre-trained model download"

    utils/hf_download.py --repo_id "unilight/sheet-models" --outdir "exp/pt_ssl-mos-wav2vec2-2337" --filename "pstn/sslmos/2337/checkpoint-19300steps.pkl"
    utils/hf_download.py --repo_id "unilight/sheet-models" --outdir "exp/pt_ssl-mos-wav2vec2-2337" --filename "pstn/sslmos/2337/config.yml"
    mv "exp/pt_ssl-mos-wav2vec2-2337/pstn/sslmos/2337/checkpoint-19300steps.pkl" "exp/pt_ssl-mos-wav2vec2-2337/checkpoint-19300steps.pkl"
    mv "exp/pt_ssl-mos-wav2vec2-2337/pstn/sslmos/2337/config.yml" "exp/pt_ssl-mos-wav2vec2-2337/config.yml"
    rm -rf "exp/pt_ssl-mos-wav2vec2-2337/pstn"

    utils/hf_download.py --repo_id "unilight/sheet-models" --outdir "exp/pt_ssl-mos-wav2vec2-3337" --filename "pstn/sslmos/3337/checkpoint-11700steps.pkl"
    utils/hf_download.py --repo_id "unilight/sheet-models" --outdir "exp/pt_ssl-mos-wav2vec2-3337" --filename "pstn/sslmos/3337/config.yml"
    mv "exp/pt_ssl-mos-wav2vec2-3337/pstn/sslmos/3337/checkpoint-11700steps.pkl" "exp/pt_ssl-mos-wav2vec2-3337/checkpoint-11700steps.pkl"
    mv "exp/pt_ssl-mos-wav2vec2-3337/pstn/sslmos/3337/config.yml" "exp/pt_ssl-mos-wav2vec2-3337/config.yml"
    rm -rf "exp/pt_ssl-mos-wav2vec2-3337/pstn"

    utils/hf_download.py --repo_id "unilight/sheet-models" --outdir "exp/pt_ssl-mos-wav2vec2-4337" --filename "pstn/sslmos/4337/checkpoint-11700steps.pkl"
    utils/hf_download.py --repo_id "unilight/sheet-models" --outdir "exp/pt_ssl-mos-wav2vec2-4337" --filename "pstn/sslmos/4337/config.yml"
    mv "exp/pt_ssl-mos-wav2vec2-4337/pstn/sslmos/4337/checkpoint-11700steps.pkl" "exp/pt_ssl-mos-wav2vec2-4337/checkpoint-11700steps.pkl"
    mv "exp/pt_ssl-mos-wav2vec2-4337/pstn/sslmos/4337/config.yml" "exp/pt_ssl-mos-wav2vec2-4337/config.yml"
    rm -rf "exp/pt_ssl-mos-wav2vec2-4337/pstn"
fi

if [ -z ${tag} ]; then
    expname="$(basename ${conf%.*})-${seed}"
else
    expname="${tag}-${seed}"
fi
expdir=exp/${expname}
if [ "${stage}" -le 2 ] && [ "${stop_stage}" -ge 2 ]; then
    echo "Stage 2: Network training"
    [ ! -e "${expdir}" ] && mkdir -p "${expdir}"
    if [ "${n_gpus}" -gt 1 ]; then
        echo "Not Implemented yet."
        # train="python -m seq2seq_vc.distributed.launch --nproc_per_node ${n_gpus} -c parallel-wavegan-train"
    else
        train="train.py"
    fi
    echo "Training start. See the progress via ${expdir}/train.log."
    ${cuda_cmd} --gpu "${n_gpus}" "${expdir}/train.log" \
        ${train} \
            --config "${conf}" \
            --train-csv-path "data/${train_set}.csv" \
            --dev-csv-path "data/${dev_set}.csv" \
            --outdir "${expdir}" \
            --resume "${resume}" \
            --verbose "${verbose}" \
            --seed "${seed}"
    echo "Successfully finished training."
fi

if [ "${stage}" -le 3 ] && [ "${stop_stage}" -ge 3 ]; then
    echo "Stage 3: Inference"
    # shellcheck disable=SC2012

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