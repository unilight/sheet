#!/usr/bin/env bash

# Copyright 2024 Wen-Chin Huang (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

conf=
datadir=

# shellcheck disable=SC1091
. utils/parse_options.sh || exit 1;

set -euo pipefail

if [ ! -z ${datadir} ]; then
    utils/BENCHMARKS/run_bvcc_test.sh --stage 1 --stop_stage 1 --conf ${conf} --datadir ${datadir}
    utils/BENCHMARKS/run_somos_test.sh --stage 1 --stop_stage 1  --conf ${conf} --datadir ${datadir}
    utils/BENCHMARKS/run_singmos_test.sh --stage 1 --stop_stage 1  --conf ${conf} --datadir ${datadir}
    utils/BENCHMARKS/run_nisqa_test.sh --stage 1 --stop_stage 1  --conf ${conf} --datadir ${datadir}
    utils/BENCHMARKS/run_tmhint_qi_test.sh --stage 1 --stop_stage 1  --conf ${conf} --datadir ${datadir}
    utils/BENCHMARKS/run_vmc23_test.sh --stage 2 --stop_stage 2  --conf ${conf}
else
    utils/BENCHMARKS/run_bvcc_test.sh --stage 1 --stop_stage 1 --conf ${conf}
    utils/BENCHMARKS/run_somos_test.sh --stage 1 --stop_stage 1  --conf ${conf}
    utils/BENCHMARKS/run_singmos_test.sh --stage 1 --stop_stage 1  --conf ${conf}
    utils/BENCHMARKS/run_nisqa_test.sh --stage 1 --stop_stage 1  --conf ${conf}
    utils/BENCHMARKS/run_tmhint_qi_test.sh --stage 1 --stop_stage 1  --conf ${conf}
    utils/BENCHMARKS/run_vmc23_test.sh --stage 1 --stop_stage 1  --conf ${conf}
fi