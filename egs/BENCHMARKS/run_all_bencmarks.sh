#!/usr/bin/env bash

# Copyright 2024 Wen-Chin Huang (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

conf=
tag=
np_inference_mode=
seed=
checkpoint=

# shellcheck disable=SC1091
. utils/parse_options.sh || exit 1;

set -euo pipefail

_opts=
if [ ! -z ${np_inference_mode} ]; then
    _opts+="--stage 3 --stop_stage 3 --np-inference-mode ${np_inference_mode} "
else
    _opts+="--stage 1 --stop_stage 1 "
fi
if [ ! -z ${tag} ]; then
    _opts+="--tag ${tag} "
fi

utils/BENCHMARKS/run_bvcc_test.sh --conf ${conf} --seed ${seed} --checkpoint ${checkpoint} ${_opts}
utils/BENCHMARKS/run_bc19_test.sh --conf ${conf} --seed ${seed} --checkpoint ${checkpoint} ${_opts}
utils/BENCHMARKS/run_somos_test.sh --conf ${conf} --seed ${seed} --checkpoint ${checkpoint} ${_opts}
utils/BENCHMARKS/run_singmos_test.sh --conf ${conf} --seed ${seed} --checkpoint ${checkpoint} ${_opts}
utils/BENCHMARKS/run_nisqa_test.sh --conf ${conf} --seed ${seed} --checkpoint ${checkpoint} ${_opts}
utils/BENCHMARKS/run_tmhint_qi_test.sh --conf ${conf} --seed ${seed} --checkpoint ${checkpoint} ${_opts}
utils/BENCHMARKS/run_vmc23_test.sh --conf ${conf} --seed ${seed} --checkpoint ${checkpoint} ${_opts}