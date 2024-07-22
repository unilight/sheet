#!/usr/bin/env bash
set -e

# Copyright 2024 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

db=$1

# download dataset
cwd=`pwd`
if [ ! -e ${db}/pstn.done ]; then
    mkdir -p ${db}
    cd ${db}
    wget https://challenge.blob.core.windows.net/pstn/train.zip
    unzip train.zip
    rm train.zip
    cd $cwd
    echo "Successfully finished download. Please follow the instructions."
    touch ${db}/pstn.done
else
    echo "Already exists. Skip download."
fi
