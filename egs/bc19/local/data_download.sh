#!/usr/bin/env bash
set -e

# Copyright 2024 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

db=$1

# download dataset
cwd=`pwd`
if [ ! -e ${db}/bc19.done ]; then
    mkdir -p ${db}
    cd ${db}
    wget https://zenodo.org/records/6572573/files/ood.tar.gz
    tar zxvf ood.tar.gz
    rm ood.tar.gz
    cd $cwd
    echo "Successfully finished download. Please follow the instructions."
    touch ${db}/bc19.done
else
    echo "Already exists. Skip download."
fi
