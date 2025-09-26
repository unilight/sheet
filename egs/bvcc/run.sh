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
meta_model_conf=conf/stacking_ridge.yaml

# dataset configuration
db_root=/data/group1/z44476r/Corpora/BVCC/main/DATA  # change this to your dataset folder

# training related setting
tag=""     # tag for directory to save model
resume=""  # checkpoint path to resume training
           # (e.g. <path>/<to>/checkpoint-10000steps.pkl)
pretrained_model_checkpoint=
datastore_path=

# decoding related setting
test_sets="bvcc_dev"
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

train_set="data/bvcc_train.csv"
dev_set="data/bvcc_dev.csv"
test_set="data/bvcc_test.csv"

mkdir -p "data"
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "stage 0: Data preparation"

    # parse original csv file to an unified format
    local/data_prep.py --generate-listener-id \
        --original-path "${db_root}/sets/TRAINSET" --wavdir "${db_root}/wav" --out "${train_set}"
    local/data_prep.py \
        --original-path "${db_root}/sets/DEVSET" --wavdir "${db_root}/wav" --out "${dev_set}"
    local/data_prep.py \
        --original-path "${db_root}/sets/TESTSET" --wavdir "${db_root}/wav" --out "${test_set}"
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "stage 1: Pre-trained model download"

    utils/hf_download.py --repo_id "unilight/sheet-models" --outdir "exp/pt_ssl-mos-wav2vec2-2337" --filename "bvcc/sslmos/2337/checkpoint-13100steps.pkl"
    utils/hf_download.py --repo_id "unilight/sheet-models" --outdir "exp/pt_ssl-mos-wav2vec2-2337" --filename "bvcc/sslmos/2337/config.yml"
    mv "exp/pt_ssl-mos-wav2vec2-2337/bvcc/sslmos/2337/checkpoint-13100steps.pkl" "exp/pt_ssl-mos-wav2vec2-2337/checkpoint-13100steps.pkl"
    mv "exp/pt_ssl-mos-wav2vec2-2337/bvcc/sslmos/2337/config.yml" "exp/pt_ssl-mos-wav2vec2-2337/config.yml"
    rm -rf "exp/pt_ssl-mos-wav2vec2-2337/bvcc"

    utils/hf_download.py --repo_id "unilight/sheet-models" --outdir "exp/pt_ssl-mos-wav2vec2-3337" --filename "bvcc/sslmos/3337/checkpoint-14900steps.pkl"
    utils/hf_download.py --repo_id "unilight/sheet-models" --outdir "exp/pt_ssl-mos-wav2vec2-3337" --filename "bvcc/sslmos/3337/config.yml"
    mv "exp/pt_ssl-mos-wav2vec2-3337/bvcc/sslmos/3337/checkpoint-14900steps.pkl" "exp/pt_ssl-mos-wav2vec2-3337/checkpoint-14900steps.pkl"
    mv "exp/pt_ssl-mos-wav2vec2-3337/bvcc/sslmos/3337/config.yml" "exp/pt_ssl-mos-wav2vec2-3337/config.yml"
    rm -rf "exp/pt_ssl-mos-wav2vec2-3337/bvcc"

    utils/hf_download.py --repo_id "unilight/sheet-models" --outdir "exp/pt_ssl-mos-wav2vec2-4337" --filename "bvcc/sslmos/4337/checkpoint-14300steps.pkl"
    utils/hf_download.py --repo_id "unilight/sheet-models" --outdir "exp/pt_ssl-mos-wav2vec2-4337" --filename "bvcc/sslmos/4337/config.yml"
    mv "exp/pt_ssl-mos-wav2vec2-4337/bvcc/sslmos/4337/checkpoint-14300steps.pkl" "exp/pt_ssl-mos-wav2vec2-4337/checkpoint-14300steps.pkl"
    mv "exp/pt_ssl-mos-wav2vec2-4337/bvcc/sslmos/4337/config.yml" "exp/pt_ssl-mos-wav2vec2-4337/config.yml"
    rm -rf "exp/pt_ssl-mos-wav2vec2-4337/bvcc"
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
            --train-csv-path "${train_set}" \
            --dev-csv-path "${dev_set}" \
            --outdir "${expdir}" \
            --resume "${resume}" \
            --verbose "${verbose}" \
            --seed "${seed}"
    echo "Successfully finished training."
fi

if [ "${stage}" -le 3 ] && [ "${stop_stage}" -ge 3 ]; then
    echo "Stage 3: Training stacking model"

    echo "Training of stacking model start. See the progress via ${expdir}/train_stack.log."
    ${cuda_cmd} --gpu "${n_gpus}" "${expdir}/train_stack.log" \
        train_stack.py \
            --meta-model-config "${meta_model_conf}" \
            --csv-path "data/bvcc_dev.csv" \
            --expdir "${expdir}" \
            --verbose "${verbose}"
    echo "Successfully finished stacking model training."

fi

if [ "${stage}" -le 4 ] && [ "${stop_stage}" -ge 4 ]; then
    echo "Stage 4: Inference"
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

###########################################################################
# Experimental ############################################################
###########################################################################

if [ "${stage}" -le 5 ] && [ "${stop_stage}" -ge 5 ]; then
    echo "Stage 5: Construct datastore"
    # shellcheck disable=SC2012

    if [ -z ${tag} ]; then
        expname="$(basename ${conf%.*})-${seed}"
    else
        expname="${tag}-${seed}"
    fi
    expdir=exp/${expname}

    [ -z "${checkpoint}" ] && checkpoint="${expdir}/checkpoint-best.pkl"
    outdir="${expdir}/datastore/$(basename "${checkpoint}" .pkl)"
    [ ! -e "${outdir}" ] && mkdir -p "${outdir}"
    [ "${n_gpus}" -gt 1 ] && n_gpus=1

    echo "Construction start. See the progress via ${outdir}/construct_datastore.log"
    ${cuda_cmd} --gpu "${n_gpus}" "${outdir}/construct_datastore.log" \
        construct_datastore.py \
            --config "${expdir}/config.yml" \
            --csv-path "data/train.csv" \
            --checkpoint "${checkpoint}" \
            --out "${outdir}/datastore.h5" \
            --verbose "${verbose}"
    echo "Successfully finished datastore construction."
fi

if [ "${stage}" -le 6 ] && [ "${stop_stage}" -ge 6 ]; then
    echo "Stage 6: Train fusion net of RAMP"

    [ ! -e "${expdir}" ] && mkdir -p "${expdir}"
    echo "Training start. See the progress via ${expdir}/train.log."

    ln -sf "$(realpath $(ls -l ${pretrained_model_checkpoint} | awk '{print $NF}'))" ${pretrained_model_checkpoint}

    pretrained_model_dir="$(dirname ${pretrained_model_checkpoint})"
    pretrained_model_checkpoint_name=$(basename ${pretrained_model_checkpoint%.*})
    # cp -uL "${pretrained_model_dir}/config.yml" "${expdir}/original_config.yml" # maybe we don't need config?
    cp -uL "${pretrained_model_checkpoint}" "${expdir}/original_${pretrained_model_checkpoint_name}.pkl"
    cp -uL "${datastore_path}" "${expdir}/datastore.h5"

    ${cuda_cmd} --gpu "${n_gpus}" "${expdir}/train.log" \
        train_ramp.py \
            --config "${conf}" \
            --train-csv-path "data/train.csv" \
            --dev-csv-path "data/dev.csv" \
            --parametric-model-checkpoint "${expdir}/original_${pretrained_model_checkpoint_name}.pkl" \
            --datastore "${expdir}/datastore.h5" \
            --outdir "${expdir}" \
            --resume "${resume}" \
            --verbose "${verbose}" \
            --seed "${seed}"

    echo "Successfully finished training."
fi
