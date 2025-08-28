#!/usr/bin/env bash
set -e

# Copyright 2025 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

db=$1

# download dataset
cwd=`pwd`
if [ ! -e ${db}/tmhint_qi_s.done ]; then
    mkdir -p ${db}
    cd ${db}

    gdown 10_1JbEsxKPYZJLDXMeMkcjLHkbDQt84w
    unzip VoiceMOS_2023_track3.zip
    rm VoiceMOS_2023_track3.zip
    mv VoiceMOS_2023_track3/* .
    rm -rf VoiceMOS_2023_track3/
    mkdir wavs
    mv *.wav wavs/
    cd ..

    cd $cwd
    echo "Successfully finished download."
    touch ${db}/tmhint_qi_s.done
else
    echo "Already exists. Skip download."
fi
