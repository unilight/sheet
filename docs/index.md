# SHEET / MOS-Bench

<p align="center">
  <img src="../assets/logo-sheet-only.png" alt="Prometheus-Logo" style="width: 30%; display: block; margin: auto;">
</p>

<b>MOS-Bench</b> is a benchmark designed to benchmark the generalization abilities of subjective speech quality assessment (SSQA) models.<br>
<b>SHEET</b> stands for the <b>S</b>peech <b>H</b>uman <b>E</b>valuation <b>E</b>stimation <b>T</b>oolkit. SHEET was designed to conduct research experiments with MOS-Bench.

## Key Features

- MOS-Bench is the first <b>large-scale collection of training and testing datasets</b> for SSQA, covering a wide range of domains, including synthetic speech from text-to-speech (TTS), voice conversion (VC), singing voice synthetis (SVS) systems, and distorted speech with artificial and real noise, clipping, transmission, reverb, etc. Researchers can use the testing sets to benchmark their SSQA model.
- This repository aims to provide **training recipes**. While there are many off-the-shelf speech quality evaluators like [DNSMOS](https://github.com/microsoft/DNS-Challenge/tree/master/DNSMOS), [SpeechMOS](https://github.com/tarepan/SpeechMOS) and [speechmetrics](https://github.com/aliutkus/speechmetrics), most of them do not provide training recipes, thus are not research-oriented. Newcomers may utilize this repo as a starting point to SSQA research.

## MOS-Bench

As of September 2025, MOS-Bench has 8 training sets and 17 test sets. See the [MOS-Bench page](mos-bench.md) for more.

## Instsallation 

### Editable installation with virtualenv 

You don't need to prepare an environment (using conda, etc.) first. The following commands will automatically construct a virtual environment in `tools/`. When you run the recipes, the scripts will automatically activate the virtual environment.

```bash
git clone https://github.com/unilight/sheet.git
cd sheet/tools
make
```

## Usage

### I just want to use your trained MOS predictor!

We utilize `torch.hub` to provide a convenient way to load pre-trained SSQA models and predict scores of wav files or torch tensors. 

!!! note

    You don't need to install sheet following the [installation instructions](#instsallation). However, you might need to install the following:

    - torch
    - h5py
    - s3prl

```python
# load pre-trained model
>>> predictor = torch.hub.load("unilight/sheet:v0.1.0", "default", trust_repo=True, force_reload=True)
# if you want to use cuda
>>> predictor.model.cuda()

# you can either provide a path to your wav file
>>> predictor.predict(wav_path="/path/to/wav/file.wav")
3.6066928

# or provide a torch tensor with shape [num_samples]
>>> predictor.predict(wav=torch.rand(16000))
1.5806346
# if you put the model on cuda...
>>> predictor.predict(wav=torch.rand(16000).cuda())
1.5806346
```

### I am new to MOS prediction research. I want to train models!

You are in the right place! This is the main purpose of SHEET. We provide complete experiment recipes, i.e., set of scripts to download and process the dataset, train and evaluate models. 

Please follow the [installation instructions](#instsallation) first, then see [training guide](training.md) for how to start.

### I already have my MOS predictor. I just want to do benchmarking!

We provide scripts to collect the test sets conveniently. These scripts can be run on Linux-like platforms with basic python requirements, such that you do not need to instal all the heavy packages, like PyTorch.

Please see [benchmarking guide](benchmarking.md) for detailed instructions.


## Supported models

### LDNet

- Original repo link: https://github.com/unilight/LDNet
- Paper link: https://arxiv.org/abs/2110.09103
- Example config: [egs/bvcc/conf/ldnet-ml.yaml](../egs/bvcc/conf/ldnet-ml.yaml)

### SSL-MOS
- Original repo link: https://github.com/nii-yamagishilab/mos-finetune-ssl
- Paper link: https://arxiv.org/abs/2110.02635
- Example config: egs/bvcc/conf/ssl-mos-wav2vec2.yaml
- Notes: We made some modifications to the original implementation. Please see the paper for more details.

### UTMOS (Strong learner)
- Original repo link: https://github.com/sarulab-speech/UTMOS22/tree/master/strong
- Paper link: https://arxiv.org/abs/2204.02152
- Example config: [egs/bvcc/conf/utmos-strong.yaml](egs/bvcc/conf/utmos-strong.yaml)

!!! note
    
    After discussion with the first author of UTMOS, Takaaki, we feel that UTMOS = SSL-MOS + listener modeling + contrastive loss + several model arch and training differences. Takaaki also felt that using phoneme and reference is not really helpful for UTMOS strong alone. Therefore we did not implement every component of UTMOS strong. For instance, we did not use domain ID and data augmentation.

### AlignNet
- Original repo link: https://github.com/NTIA/alignnet
- Paper link: https://arxiv.org/abs/22406.10205
- Example config: egs/bvcc+nisqa+pstn+singmos+somos+tencent+tmhint-qi/conf/alignnet-wav2vec2.yaml

## Supported features

### Modeling

- Listener modeling
- Self-supervised learning (SSL) based encoder, supported by S3PRL

!!! note

    Find the complete list of supported SSL models [here](https://s3prl.github.io/s3prl/tutorial/upstream_collection.html)

### Training

- Automatic best-n model saving and early stopiing based on given validation criterion
- Visualization, including predicted score distribution, scatter plot of utterance and system level scores
- Model averaging
- Model ensembling by stacking