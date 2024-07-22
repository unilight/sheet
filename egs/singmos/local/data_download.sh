#!/usr/bin/env bash
set -e

# Copyright 2024 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

db=$1

# download dataset
cwd=`pwd`
if [ ! -e ${db}/singmos.done ]; then
    mkdir -p ${db}
    cd ${db}
    gdown 1DtzZhk3M_jsxUxirPcFRoBhq-dsinOWN
    gdown 1sO4xPUMJvGAjC8lmO6uXCwgz7s7Ruhpv
    unzip voicemos2024-track2-train-phase.zip
    unzip voicemos2024-track2-eval-phase.zip
    rm voicemos2024-track2-train-phase.zip
    rm voicemos2024-track2-eval-phase.zip
    cd $cwd
    echo "Successfully finished download."
    touch ${db}/singmos.done
else
    echo "Already exists. Skip download."
fi
