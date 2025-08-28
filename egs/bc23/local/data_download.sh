#!/usr/bin/env bash
set -e

# Copyright 2025 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

db=$1

# download dataset
cwd=`pwd`
if [ ! -e ${db}/bc23.done ]; then
    mkdir -p ${db}
    cd ${db}

    wget https://www.dropbox.com/s/c83l67bkeh9p49k/VoiceMOS2023Track1.zip
    unzip VoiceMOS2023Track1.zip
    rm VoiceMOS2023Track1.zip
    rm -rf __MACOSX/
    mv VoiceMOS2023Track1/* .
    rm -rf VoiceMOS2023Track1/
    mkdir wavs
    mv *.wav wavs/
    cd ..

    cd $cwd
    echo "Successfully finished download."
    touch ${db}/bc23.done
else
    echo "Already exists. Skip download."
fi
