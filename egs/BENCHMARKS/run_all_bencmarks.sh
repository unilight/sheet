#!/usr/bin/env bash

# Copyright 2024 Wen-Chin Huang (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

conf=
np_inference_mode=
seed=

# shellcheck disable=SC1091
. utils/parse_options.sh || exit 1;

set -euo pipefail

if [ ! -z ${np_inference_mode} ]; then
    utils/BENCHMARKS/run_bvcc_test.sh --stage 3 --stop_stage 3 --conf ${conf} --seed ${seed} --np-inference-mode "${np_inference_mode}"
    utils/BENCHMARKS/run_bc19_test.sh --stage 3 --stop_stage 3 --conf ${conf} --seed ${seed} --np-inference-mode "${np_inference_mode}"
    utils/BENCHMARKS/run_somos_test.sh --stage 3 --stop_stage 3  --conf ${conf} --seed ${seed} --np-inference-mode "${np_inference_mode}"
    utils/BENCHMARKS/run_singmos_test.sh --stage 3 --stop_stage 3  --conf ${conf} --seed ${seed} --np-inference-mode "${np_inference_mode}"
    utils/BENCHMARKS/run_nisqa_test.sh --stage 3 --stop_stage 3  --conf ${conf} --seed ${seed} --np-inference-mode "${np_inference_mode}"
    utils/BENCHMARKS/run_tmhint_qi_test.sh --stage 3 --stop_stage 3  --conf ${conf} --seed ${seed} --np-inference-mode "${np_inference_mode}"
    utils/BENCHMARKS/run_vmc23_test.sh --stage 3 --stop_stage 3  --conf ${conf} --seed ${seed} --np-inference-mode "${np_inference_mode}"
else
    utils/BENCHMARKS/run_bvcc_test.sh --stage 1 --stop_stage 1 --conf ${conf} --seed ${seed}
    utils/BENCHMARKS/run_bc19_test.sh --stage 1 --stop_stage 1 --conf ${conf} --seed ${seed}
    utils/BENCHMARKS/run_somos_test.sh --stage 1 --stop_stage 1  --conf ${conf} --seed ${seed}
    utils/BENCHMARKS/run_singmos_test.sh --stage 1 --stop_stage 1  --conf ${conf} --seed ${seed}
    utils/BENCHMARKS/run_nisqa_test.sh --stage 1 --stop_stage 1  --conf ${conf} --seed ${seed}
    utils/BENCHMARKS/run_tmhint_qi_test.sh --stage 1 --stop_stage 1  --conf ${conf} --seed ${seed}
    utils/BENCHMARKS/run_vmc23_test.sh --stage 1 --stop_stage 1  --conf ${conf} --seed ${seed}
fi