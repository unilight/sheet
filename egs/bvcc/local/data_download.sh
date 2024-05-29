#!/usr/bin/env bash
set -e

# Copyright 2024 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

db=$1

# download dataset
cwd=`pwd`
if [ ! -e ${db}/main.done ]; then
    mkdir -p ${db}
    cd ${db}
    wget https://zenodo.org/records/6572573/files/main.tar.gz
    tar zxvf main.tar.gz
    rm main.tar.gz
    cd $cwd
    echo "Successfully finished download. Please follow the instructions."
    touch ${db}/main.done
else
    echo "Already exists. Skip download."
fi
