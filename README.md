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

## Usage

Same as many speech processing based repositories ([ESPNet](https://github.com/espnet/espnet), [ParallelWaveGAN](https://github.com/kan-bayashi/ParallelWaveGAN), etc.), we formulate our recipes in kaldi-style. They can be found in the `egs` folder.

There are several usages of this toolkit.

### Training a speech quality predictor with a supported dataset on your own

You can train your own speech quality predictor using the datasets we support. Usually these datasets come with their own test sets, and you can test on these sets after training finishes. The starting point of each recipe is the `run.sh` file. Please check the detailed usage in each recipe.

### Zero-shot prediction on multiple benchmarks with your trained model

Inside each recipe, after the model training is done, you can run zero-shot prediction on other datasets and benchmarks. This can be done by running `run_XXX_test.sh` in each recipe. They are symbolic links to the scripts in the `egs/BENCHMARKS` folder.

## Supported Training datasets

Currently we support (and have tested) training recipes on the following datasets:

- BVCC
    - Dataset download link: https://zenodo.org/records/6572573
    - Paper link: [[Original paper](https://arxiv.org/abs/2105.02373)] [[VoiceMOS Challenge 2022](https://arxiv.org/abs/2203.11389)]
    - Recipe: `egs/bvcc`
- SOMOS
    - Dataset download link: https://zenodo.org/records/7378801
    - Paper link: [[arXiv version](https://arxiv.org/abs/2204.03040)]
    - Recipe: `egs/somos`
- NISQA
    - Dataset download link: https://github.com/gabrielmittag/NISQA/wiki/NISQA-Corpus
    - Paper link: [[arXiv version](https://arxiv.org/abs/2104.09494)]
    - Recipe: `egs/nisqa`

## Supported Benchmarks

In addition to the test sets provided in the datasets above, you can do zero-shot evaluation on the following benchmarks.
For usage, see [egs/BENCHMARK/README.md](egs/BENCHMARK)

- VoiceMOS Challenge 2023 (BC2023, SVCC2023, TMHINTQI-(S))
    - Paper link: [[arXiv version](https://arxiv.org/abs/2310.02640)]
    - Recipe: `egs/BENCHMARK/run_vmc23_test.sh`

Of course you can also do zero-shot benchmarking on the test sets of the training datasets:

- BVCC: `egs/BENCHMARK/run_bvcc_test.sh`
- SOMOS: `egs/BENCHMARK/run_somos_test.sh`
- NISQA: `egs/BENCHMARK/run_nisqa_test.sh`
    
In the future, we plan to support the following additional benchmarks:

- VoiceMOS Challenge 2022 OOD track
- VoiceMOS Challenge 2024

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
- UTMOS (Strong learner)
    - Original repo link: https://github.com/sarulab-speech/UTMOS22/tree/master/strong
    - Paper link: [[arXiv](https://arxiv.org/abs/2204.02152)]
    - Example config: `egs/bvcc/conf/utmos-strong.yaml`
    - Notes:
        - After discussion with the first author of UTMOS, Takaaki, we feel that UTMOS = SSL-MOS + listener modeling + contrastive loss + several model arch and training differences. Takaaki also felt that using phoneme and reference is not really helpful for UTMOS strong alone. Therefore we did not implement every component of UTMOS strong. For instance, we did not use domain ID and data augmentation.


## Supported Features

- Modeling
    - Listener modeling
    - Self-supervised learning (SSL) based encoder, supported by S3PRL
      - Find the complete list of supported SSL models [here](https://s3prl.github.io/s3prl/tutorial/upstream_collection.html).
- Training
    - Automatic best-n model saving and early stopiing based on given validation criterion
    - Visualization, including predicted score distribution, scatter plot of utterance and system level scores
    - Model averaging
    - Model ensembling by stacking


## Acknowledgements

This repo is greatly inspired by the following repos. Or I should say, many code snippets are directly taken from part of the following repos.

- [ESPNet](https://github.com/espnet/espnet)
- [ParallelWaveGAN](https://github.com/kan-bayashi/ParallelWaveGAN/)

## Author

Wen-Chin Huang  
Toda Labotorary, Nagoya University  
E-mail: wen.chinhuang@g.sp.m.is.nagoya-u.ac.jp