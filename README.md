# 📃 SHEET: Speech Human Evaluation Estimation Toolkit

## Introduction and motivation

Recently, predicting human ratings of speech using data-driven models (specifically, deep neural networks) has been a popular research topic. This repository aims to provide **training recipes** to reproduce some popular models.

## Instsallation 

### Editable installation with virtualenv 

```
git clone https://github.com/unilight/sheet.git
cd sheet/tools
make
```

## Complete training, decoding and benchmarking

Same as many speech processing based repositories ([ESPNet](https://github.com/espnet/espnet), [ParallelWaveGAN](https://github.com/kan-bayashi/ParallelWaveGAN), etc.), we formulate our recipes in kaldi-style. They can be found in the `egs` folder. Please check the detailed usage in each recipe.

## Supported Datasets

Currently we support the following datasets:

- BVCC
    - Dataset download link: https://zenodo.org/records/6572573
    - Paper link: [[Original paper](https://arxiv.org/abs/2105.02373)] [[VoiceMOS Challenge 2022](https://arxiv.org/abs/2203.11389)]
    - Recipe: `egs/bvcc`

We plan to support the following datasets in the future:

- SOMOS
- NISQA

## Supported Models

Currently we support the following models:

- LDNet
    - Original repo link: https://github.com/unilight/LDNet
    - Paper link: [[arXiv](https://arxiv.org/abs/2110.09103)]
    - Example config: `egs/bvcc/conf/ldnet-ml.yaml`
- SSL-MOS
    - Original repo link: https://github.com/nii-yamagishilab/mos-finetune-ssl/tree/main
    - Paper link: [[arXiv](https://arxiv.org/abs/2110.02635)]
    - Example config: `egs/bvcc/conf/ssl-mos-wav2vec2.yaml`

We plan to support the following models in the future:

- UTMOS

or, support some model that mixes core techniques in the models above.

## Supported Features

- Modeling
    - Listener modeling
    - Self-supervised learning (SSL) based encoder
- Training
    - Automatic best-n model saving and early stopiing based on given validation criterion
    - Visualization, including predicted score distribution, scatter plot of utterance and system level scores


## Acknowledgements

This repo is greatly inspired by the following repos. Or I should say, many code snippets are directly taken from part of the following repos.

- [ESPNet](https://github.com/espnet/espnet)
- [ParallelWaveGAN](https://github.com/kan-bayashi/ParallelWaveGAN/)

## Author

Wen-Chin Huang  
Toda Labotorary, Nagoya University  
E-mail: wen.chinhuang@g.sp.m.is.nagoya-u.ac.jp