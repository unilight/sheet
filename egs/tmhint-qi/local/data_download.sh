#!/usr/bin/env bash
set -e

# Copyright 2024 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

db=$1

# download dataset
cwd=`pwd`
if [ ! -e ${db}/tmhint-qi.done ]; then
    mkdir -p ${db}
    cd ${db}
    gdown 1TMDiz6dnS76hxyeAcCQxeSqqEOH4UDN0
    unzip TMHINTQI.zip
    rm TMHINTQI.zip
    rm -rf __MACOSX/
    mv TMHINTQI/* .
    rm -rf TMHINTQI
    cd $cwd
    echo "Successfully finished download. Please follow the instructions."
    touch ${db}/main.done
else
    echo "Already exists. Skip download."
fi
