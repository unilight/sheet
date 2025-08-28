#!/usr/bin/env bash
set -e

# Copyright 2025 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

db=$1

# download dataset
cwd=`pwd`
if [ ! -e ${db}/svcc23.done ]; then
    mkdir -p ${db}
    cd ${db}

    gdown 188FJiCBT0RSI6-q4ICJPfGqmR22_9Kd2
    unzip VoiceMOS_2023_track2.zip
    rm VoiceMOS_2023_track2.zip
    mv VoiceMOS_2023_track2/* .
    rm -rf VoiceMOS_2023_track2/
    mkdir wavs
    mv *.wav wavs/
    cd ..

    cd $cwd
    echo "Successfully finished download."
    touch ${db}/svcc23.done
else
    echo "Already exists. Skip download."
fi
