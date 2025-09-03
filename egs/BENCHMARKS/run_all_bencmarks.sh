#!/usr/bin/env bash

# Copyright 2025 Wen-Chin Huang (Nagoya University)
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

# synthetic speech
utils/BENCHMARKS/run_bvcc_test.sh --conf ${conf} --seed ${seed} --checkpoint ${checkpoint} ${_opts}
utils/BENCHMARKS/run_somos_test.sh --conf ${conf} --seed ${seed} --checkpoint ${checkpoint} ${_opts}
utils/BENCHMARKS/run_bc19_test.sh --conf ${conf} --seed ${seed} --checkpoint ${checkpoint} ${_opts}
utils/BENCHMARKS/run_bc23_test.sh --conf ${conf} --seed ${seed} --checkpoint ${checkpoint} ${_opts}
utils/BENCHMARKS/run_svcc23_test.sh --conf ${conf} --seed ${seed} --checkpoint ${checkpoint} ${_opts}
utils/BENCHMARKS/run_singmos_test.sh --conf ${conf} --seed ${seed} --checkpoint ${checkpoint} ${_opts}
utils/BENCHMARKS/run_brspeechmos_test.sh --conf ${conf} --seed ${seed} --checkpoint ${checkpoint} ${_opts}
utils/BENCHMARKS/run_hablamos_test.sh --conf ${conf} --seed ${seed} --checkpoint ${checkpoint} ${_opts}
utils/BENCHMARKS/run_ttsds2_test.sh --conf ${conf} --seed ${seed} --checkpoint ${checkpoint} ${_opts}

# distorted speech
utils/BENCHMARKS/run_nisqa_test.sh --conf ${conf} --seed ${seed} --checkpoint ${checkpoint} ${_opts}
utils/BENCHMARKS/run_tmhint_qi_test.sh --conf ${conf} --seed ${seed} --checkpoint ${checkpoint} ${_opts}
utils/BENCHMARKS/run_tmhint_qi_s_test.sh --conf ${conf} --seed ${seed} --checkpoint ${checkpoint} ${_opts}
utils/BENCHMARKS/run_tcd_voip_test.sh --conf ${conf} --seed ${seed} --checkpoint ${checkpoint} ${_opts}
utils/BENCHMARKS/run_vmc24_track3_test.sh --conf ${conf} --seed ${seed} --checkpoint ${checkpoint} ${_opts}