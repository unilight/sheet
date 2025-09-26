#!/usr/bin/env bash

# Copyright 2024 Wen-Chin Huang (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)
# modiied from https://github.com/soumimaiti/speechlmscore_tool/tree/main

# basic settings
stage=-1       # stage to start
stop_stage=100 # stage to stop
n_gpus=1       # number of gpus

pretrained_model_dir=
csv_path=
outdir=

hubert_layer_idx=5

# shellcheck disable=SC1091
. utils/parse_options.sh || exit 1;

set -euo pipefail


if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    echo "stage -1: Pretrained Model Download"

    mkdir -p ${pretrained_model_dir}
    if [ ! -e ${pretrained_model_dir}/hubert_base_ls960.done ]; then
        wget https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt -P "${pretrained_model_dir}"
        touch ${pretrained_model_dir}/hubert_base_ls960.done
    else
        echo "Already exists. Skip download."
    fi

    if [ ! -e ${pretrained_model_dir}/km.done ]; then
        wget https://dl.fbaipublicfiles.com/textless_nlp/gslm/hubert/km50/km.bin -P "${pretrained_model_dir}"
        touch ${pretrained_model_dir}/km.done
    else
        echo "Already exists. Skip download."
    fi

    if [ ! -e ${pretrained_model_dir}/ulm.done ]; then
        utils/hf_download.py --repo_id "soumi-maiti/speech-ulm-lstm" --outdir "${pretrained_model_dir}" --filename "tokens.txt"
        utils/hf_download.py --repo_id "soumi-maiti/speech-ulm-lstm" --outdir "${pretrained_model_dir}" --filename "config.yaml"
        utils/hf_download.py --repo_id "soumi-maiti/speech-ulm-lstm" --outdir "${pretrained_model_dir}" --filename "valid.loss.best.pth"
        touch ${pretrained_model_dir}/ulm.done
    else
        echo "Already exists. Skip download."
    fi

fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "stage 0: Evaluate"

    ${cuda_cmd} --gpu "${n_gpus}" "${outdir}/evaluate.log" \
        speechlmscore_evaluate.py \
            --csv-path "${csv_path}" \
            --km-model-path "${pretrained_model_dir}/km.bin" \
            --layer "${hubert_layer_idx}" \
            --ulm-token-list "${pretrained_model_dir}/tokens.txt" \
            --ulm-config "${pretrained_model_dir}/config.yaml" \
            --ulm-model-path "${pretrained_model_dir}/valid.loss.best.pth" \
            --outdir "${outdir}"
fi

