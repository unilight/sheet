#!/usr/bin/env bash
set -e

# Copyright 2025 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

db=$1

# download dataset
cwd=`pwd`
if [ ! -e ${db}/hablamos.done ]; then
    mkdir -p ${db}
    cd ${db}

    wget https://zenodo.org/records/17024887/files/HablaMOS.tar.gz
    tar zxcf HablaMOS.tar.gz
    rm -f HablaMOS.tar.gz
    mv HablaMOS/* .
    rm -rf HablaMOS/

    cd $cwd
    echo "Successfully finished download."
    touch ${db}/hablamos.done
else
    echo "Already exists. Skip download."
fi
