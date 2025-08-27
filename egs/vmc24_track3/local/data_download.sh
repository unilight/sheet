#!/usr/bin/env bash
set -e

# Copyright 2025 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

db=$1

# download dataset
cwd=`pwd`
if [ ! -e ${db}/vmc24_track3.done ]; then
    mkdir -p ${db}
    cd ${db}

    gdown 10ZPEccntb_KthiPYQhgCBjvyytkckAXt
    unzip voicemos2024-track3-eval-phase.zip
    rm -f voicemos2024-track3-eval-phase.zip

    cd $cwd
    echo "Successfully finished download."
    touch ${db}/vmc24_track3.done
else
    echo "Already exists. Skip download."
fi
