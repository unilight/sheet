#!/usr/bin/env bash
set -e

# Copyright 2024 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

db=$1

# download dataset
cwd=`pwd`
if [ ! -e ${db}/somos.done ]; then
    mkdir -p ${db}
    cd ${db}
    wget https://zenodo.org/records/7378801/files/somos.zip
    unzip somos.zip
    unzip audios.zip
    rm somos.zip
    cd $cwd
    echo "Successfully finished download."
    touch ${db}/somos.done
else
    echo "Already exists. Skip download."
fi
