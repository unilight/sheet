<p align="center">
  <img src="assets/logo-sheet-only.png" alt="Prometheus-Logo" style="width: 30%; display: block; margin: auto;">
</p>

<h1 align="center">🗣️ SHEET / MOS-Bench 🎧 </h1>
<h3 align="center">Manipulate MOS-Bench with SHEET</h3>


- <b>MOS-Bench</b> is a benchmark designed to benchmark the generalization abilities of subjective speech quality assessment (SSQA) models.
- <b>SHEET</b> stands for the <b>S</b>peech <b>H</b>uman <b>E</b>valuation <b>E</b>stimation <b>T</b>oolkit. SHEET was designed to conduct research experiments with MOS-Bench.

📚 **[Full Documentation (NEW!)](https://unilight.github.io/sheet/)** | 📝 **[arXiv paper(2024)](https://arxiv.org/abs/2411.03715)** | 🤗 **[HuggingFace Space demo](https://huggingface.co/spaces/unilight/sheet-demo)**

## MOS-Bench Overview

See this **[Google Spreadsheet](https://docs.google.com/spreadsheets/d/1Uqi6upfJHasoduuY72_75qphJgU-PuY-u3OKBBMNaKI/edit?usp=sharing)** for an overview of the datasets in MOS-Bench.

- **Sep 2025**: MOS-Bench now has **8 training sets and 17 test sets**. 
- <span style="color: grey;">Nov 2024: The initial MOS-Bench has 7 training sets and 12 test sets.</span>

## Usage guide

There are three usages of SHEET:
- I am new to MOS prediction research. I want to train models! → [Training guide]()
- I already have my MOS predictor. I just want to do benchmarking!  → [Benchmarking guide](#)
- I just want to use your trained MOS predictor!  → [Quick start](#quick-start)

## Quick start

We utilize `torch.hub` to provide a convenient way to load pre-trained SSQA models and predict scores of wav files or torch tensors.

[!NOTE]
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

## Instsallation 

### Editable installation with virtualenv 

You don't need to prepare an environment (using conda, etc.) first. The following commands will automatically construct a virtual environment in `tools/`. When you run the recipes, the scripts will automatically activate the virtual environment.

```bash
git clone https://github.com/unilight/sheet.git
cd sheet/tools
make
```




## Information

### Citation

If you use the training scripts, benchmarking scripts or pre-trained models from this project, please consider citing the following paper.

```
@inproceedings{sheet,
  title     = {{SHEET: A Multi-purpose Open-source Speech Human Evaluation Estimation Toolkit}},
  author    = {Wen-Chin Huang and Erica Cooper and Tomoki Toda},
  year      = {2025},
  booktitle = {{Proc. Interspeech}},
  pages     = {2355--2359},
}


@article{huang2024,
      title={MOS-Bench: Benchmarking Generalization Abilities of Subjective Speech Quality Assessment Models}, 
      author={Wen-Chin Huang and Erica Cooper and Tomoki Toda},
      year={2024},
      eprint={2411.03715},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2411.03715}, 
}
```

### Acknowledgements

This repo is greatly inspired by the following repos. Or I should say, many code snippets are directly taken from part of the following repos.

- [ESPNet](https://github.com/espnet/espnet)
- [ParallelWaveGAN](https://github.com/kan-bayashi/ParallelWaveGAN/)

### Author

Wen-Chin Huang  
Toda Labotorary, Nagoya University  
E-mail: wen.chinhuang@g.sp.m.is.nagoya-u.ac.jp
