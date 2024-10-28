#!/usr/bin/env bash
set -e

# Copyright 2024 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

db=$1

# download dataset
cwd=`pwd`
if [ ! -e ${db}/nisqa.done ]; then
    mkdir -p ${db}
    cd ${db}
    wget https://depositonce.tu-berlin.de/bitstream/11303/13012.5/9/NISQA_Corpus.zip
    unzip NISQA_Corpus.zip
    rm -f NISQA_Corpus.zip
    mv NISQA_Corpus/* .
    rm -rf NISQA_Corpus/
    cd $cwd
    echo "Successfully finished download."
    touch ${db}/nisqa.done
else
    echo "Already exists. Skip download."
fi
