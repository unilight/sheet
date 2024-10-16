#!/usr/bin/env bash

# Copyright 2024 Wen-Chin Huang (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

stage=-1       # stage to start
stop_stage=0   # stage to stop

# shellcheck disable=SC1091
. utils/parse_options.sh || exit 1;

set -euo pipefail

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    echo "stage -1: Download data for all benchmark sets"

    _opts+="--stage -1 --stop_stage -1 "

    utils/BENCHMARKS/run_bvcc_test.sh ${_opts}
    utils/BENCHMARKS/run_bc19_test.sh ${_opts}
    utils/BENCHMARKS/run_somos_test.sh ${_opts}
    utils/BENCHMARKS/run_singmos_test.sh ${_opts}
    utils/BENCHMARKS/run_nisqa_test.sh ${_opts}
    utils/BENCHMARKS/run_tmhint_qi_test.sh ${_opts}
    utils/BENCHMARKS/run_vmc23_test.sh ${_opts}

    echo "Please follow instructions in bvcc, bc19 to finish the download process."
fi


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "stage 0: Data preparation for all benchmark sets"

    _opts+="--stage 0 --stop_stage 0 "

    utils/BENCHMARKS/run_bvcc_test.sh ${_opts}
    utils/BENCHMARKS/run_bc19_test.sh ${_opts}
    utils/BENCHMARKS/run_somos_test.sh ${_opts}
    utils/BENCHMARKS/run_singmos_test.sh ${_opts}
    utils/BENCHMARKS/run_nisqa_test.sh ${_opts}
    utils/BENCHMARKS/run_tmhint_qi_test.sh ${_opts}
    utils/BENCHMARKS/run_vmc23_test.sh ${_opts}
fi
