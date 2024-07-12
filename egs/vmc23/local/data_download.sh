#!/usr/bin/env bash
set -e

# Copyright 2024 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

db=$1

# download dataset
cwd=`pwd`
if [ ! -e ${db}/vmc23.done ]; then
    mkdir -p ${db}
    cd ${db}
    mkdir track1
    mkdir track2
    mkdir track3

    # track 1
    cd track1
    wget https://www.dropbox.com/s/c83l67bkeh9p49k/VoiceMOS2023Track1.zip
    unzip VoiceMOS2023Track1.zip
    rm VoiceMOS2023Track1.zip
    rm -rf __MACOSX/
    mv VoiceMOS2023Track1/* .
    rm -rf VoiceMOS2023Track1/
    cd ..

    # track 2
    cd track2
    gdown 188FJiCBT0RSI6-q4ICJPfGqmR22_9Kd2
    unzip VoiceMOS_2023_track2.zip
    rm VoiceMOS_2023_track2.zip
    mv VoiceMOS_2023_track2/* .
    rm -rf VoiceMOS_2023_track2/
    cd ..

    # track 3
    cd track3
    gdown 10_1JbEsxKPYZJLDXMeMkcjLHkbDQt84w
    unzip VoiceMOS_2023_track3.zip
    rm VoiceMOS_2023_track3.zip
    mv VoiceMOS_2023_track3/* .
    rm -rf VoiceMOS_2023_track3/
    cd ..

    cd $cwd
    echo "Successfully finished download."
    touch ${db}/vmc23.done
else
    echo "Already exists. Skip download."
fi
