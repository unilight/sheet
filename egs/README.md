# egs (examples, or recipes)

## Table of Contents

- [Usage](#usage)
  - [Conduct complete experiments (training & benchmarking)](#conduct-complete-experiments-training--benchmarking)
  - [Reproduce]
  - [Get the test sets in MOS-Bench](#get-the-test-sets-in-mos-bench)
- [Training datasets in MOS-Bench](#training-datasets-in-mos-bench)
- [Testing datasets in MOS-Bench](#testing-sets-in-mos-bench)

## Usage

### Conduct complete experiments (training & benchmarking)

You can train your own speech quality predictor using the datasets in MOS-Bench. Each The starting point of each recipe is the `run.sh` file.

There are two ways you can train models:
- Train models using a single dataset at a time. This is called "**single dataset training**". You can refer to the [Training datasets in MOS-Bench](#training-datasets-in-mos-bench) section for available datasets.
- Train models by pooling multiple datasets. This is called "**multiple dataset training**". Currently we support the following recipe:
    - Recipe: `egs/bvcc+nisqa+pstn+singmos+somos+tencent+tmhint-qi`

After you train your models, you can run benchmarking on the test sets in MOS-Bench.

Below, we explain the processes of [training](#structure-of-each-training-recipe-runsh) and [benchmarking](#structure-of-each-benchmark-recipe-benchmarksrun_xxx_testsh).

#### Structure of each training recipe (`run.sh`)

For most recipes, by default the SSL-MOS model is used. The configuration file is `conf/ssl-mos-wav2vec2.yaml`.

##### Data download

```bash
./run.sh --stage -1 --stop_stage -1
```

By default, the dataset will be put in the `downloads` folder in each recipe folder. However, you can change the destination to your favor.

For most datasets, this step automatically calls the `local/data_download.sh` script to fetch the dataset. However, some datasets (like `bvcc`) does not come with an one-step automatic download script due to the data policy. In that case, please follow the instructions.

##### Data preparation

```bash
./run.sh --stage 0 --stop_stage 0
```

This step processes the downloaded dataset and generate `.csv` files for the training and development (and possibly testing) sets. The generated data files are in `data`. For instance, `bvcc/data/bvcc_<train, dev, test>.csv` files will be generated.

The common data format across all recipes is csv. Each csv file always contains the following columns:

- `wav_path`: **Absolute** paths to the wav files.
- `score`: This is the listener-dependent score by default. If not available, then it will be the MOS.
- `system_id`: System ID. This is usually for synthetic datasets. For some datasets, this is set to a dump ID.
- `sample_id`: Sample ID. Should be unique within one single dataset.

Optionally, the following columns may exist:

- `listener_id`: The original listener ID in the dataset. Only when listener ID is available in the dataset.
- `listener_idx` : A listener **INDEX** for the listener ID. This is an integer scalar.
- `phoneme`: This only exists for BVCC. Not really important.
- `cluster`: This only exists for BVCC. Not really important.
- `reference`: This only exists for BVCC. Not really important.

_Note: stage 1 is deprecated and not used._

##### Training

```
./run.sh --stage 2 --stop_stage 2 \
    --conf <conf/config.yml> --tag <tag> --seed <seed> # these are optionals
```

Training is launched within this stage. All generated artifacts will be saved to a `expdir`, which is by default `exp/ssl-mos-wav2vec2-1337`. The `1337` is the random seed set by `--seed`. If `--tag` is specified, then they will be saved in `exp/<tag>-1337`. The model checkpoints can be found in the expdir.  Also, you can check the `exp/ssl-mos-wav2vec2-1337/intermediate_results` to see some plots to monitor the training process.

#### Structure of each benchmark recipe (`BENCHMARKS/run_XXX_test.sh`)

After the [training process described above](#structure-of-each-training-recipe-runsh), **INSIDE THOSE RECIPES** you can follow the commands below to do benchmarking.

As an example, we assume we trained a model in `egs/bvcc`, and we want to run inference with the BVCC test set. That is, we will be executing `run_bvcc_test.sh`.

##### Data download & preparation

```bash
# data download.
# for certain datasets (ex., BVCC or BC19), folow the instructions to finish downloading.
./utils/BENCHMARKS/run_bvcc_test.sh --stage -1 --stop_stage -1

# data preparation
./utils/BENCHMARKS/run_bvcc_test.sh --stage 0 --stop_stage 0
```

The purpose of this stage is the same as the data download & preparation stage as [described in the training process](#structure-of-each-training-recipe-runsh). The generated data csv files will be stored in the `data` folder **IN EACH CORRESPONDING TEST SET RECIPE FOLDER**. For instance, for `run_bvcc_test.sh`, the generated data csv files will be in `bvcc/data/bvcc_<dev, test>.sh`. This also holds for test sets without training recipes: for instance, `run_vmc23_test.sh` will generate data csv files to `vmc23/data/vmc23_track<1a, 1b, 2, 3>_test`.

##### Inference

The following command runs **parametric inference**.

```bash
./utils/BENCHMARKS/run_bvcc_test.sh --stage 1 --stop_stage 1 \\
    --conf <conf/config.yml> --checkpoint <checkpoint file> --tag <tag> # these are optionals
```

Inference is done within this stage. The results will be saved in `exp/<expdir>/results/<checkpoint_name>/<test_set_name>/`. For instance, `exp/ssl-mos-wav2vec2-1337/results/checkpoint-best/bvcc_test`. Inside you can find the following files:
- `inference.log`: log file of the inference script, along with the calculated metrics.
- `distribution.png`: distribution over the score range (1-5).
- `utt_scatter_plot.png`: utterance-wise scatter plot of the ground truth and the predicted scores.
- `sys_scatter_plot.png`: system-wise scatter plot of the ground truth and the predicted scores.

By default, the `checkpoint-best.pkl` is used, which is a symbolic link that points to the best performing model checkpoint (depending on the `best_model_criterion` field in the config file.) You can specify a different checkpoint file with `--checkpoint`.

###### Non-parametric inference

You can also run **non-parametric inference** for certain models. However, not all recipes and models support it.
- Note: as of 2024.10, currently, only the `egs/bvcc+nisqa+pstn+singmos+somos+tencent+tmhint-qi` recipe supports non-parametric inference. However, it is not difficult to add it to other recipes. It's just that we haven't testes it yet.

If you want to run non-parametric inference, you need to prepare the datastore first. This can be done by:

```bash
./run.sh --stage 3 --stop_stage 3 \
    --conf <conf/config.yml> --tag <tag> --seed <seed> # these are optionals
```

Then, you can do non-parametric inference with the following command:

```bash
./utils/BENCHMARKS/run_bvcc_test.sh --stage 3 --stop_stage 3 \
    --np_inference_mode <naive_knn/domain_id_knn_1> \
    --conf <conf/config.yml> --checkpoint --tag <tag> # these are optionals
```

Note that the results will be then stored in `exp/<expdir>/results/np_<checkpoint_name>/<np_inference_mode>/<test_set_name>/`. For instance, `egs/bvcc+nisqa+pstn+singmos+somos+tencent+tmhint-qi/exp/alignnet-wav2vec2-2337/results/np_checkpoint-best/naive_knn/bc19_test`.

###### Run all benchmarks at once

You can also run all benchmarks at once. First, download and prepare all the benchmark sets.

```bash
# data download.
# for certain datasets (ex., BVCC or BC19), folow the instructions to finish downloading.
./utils/BENCHMARKS/get_all_bencmarks.sh --stage -1 --stop_stage -1

# data preparation
./utils/BENCHMARKS/get_all_bencmarks.sh --stage 0 --stop_stage 0
```

Then, run inference based on the mode:
- Parametric inference:
  ```bash
  ./utils/BENCHMARKS/run_all_bencmarks.sh \
    --conf conf/ssl-mos-wav2vec2 --checkpoint <checkpoint> --tag <tag> --seed <seed>  # these are optionals
  ```
- Non-parametric inference:

  ```bash
  ./utils/BENCHMARKS/run_all_bencmarks.sh --np_inference_mode <naive_knn/domain_id_knn_1> \
    --conf conf/ssl-mos-wav2vec2 --checkpoint <checkpoint> --tag <tag> --seed <seed>  # these are optionals
  ```

### Download pre-trained SSQA models to reproduce the paper results

All the models are hosted on HuggingFace Models. <a href="https://huggingface.co/unilight/sheet-models"><img src="https://img.shields.io/badge/HuggingFace%20Models-FFD21E?logo=huggingface&logoColor=000" alt="Prometheus-Logo" style="width: 25%"></a>

To 

### Get the test sets in MOS-Bench

This section is for users who don't want to train SSQA models but just want to get the test sets.
The design is that **you don't need to run the installation, which can be time-consuming.**

First, please set the following two variables in `egs/BENCHMARKS/get_all_bencmarks_installation_free.sh`:
- `db_root`: the root directory to save the raw datsets, including the wav files.
- `datadir`: the directory to save the `csv` files.

```bash
cd egs/BENCHMARKS

# data download.
# for certain datasets (ex., BVCC or BC19), folow the instructions to finish downloading.
./utils/BENCHMARKS/get_all_bencmarks_installation_free.sh --stage -1 --stop_stage -1

# data preparation
./utils/BENCHMARKS/get_all_bencmarks_installation_free.sh --stage 0 --stop_stage 0
```

## Training datasets in MOS-Bench

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
- TMHINT-QI
    - Dataset download link: https://drive.google.com/file/d/1TMDiz6dnS76hxyeAcCQxeSqqEOH4UDN0/view?usp=sharing
    - Paper link: [[INTERSPEECH 2022 version](https://www.isca-speech.org/archive/pdfs/interspeech_2022/chen22i_interspeech.pdf)]
    - Recipe: `egs/tmhint-qi`
- PSTN
    - Dataset download link: https://challenge.blob.core.windows.net/pstn/train.zip
    - Paper link: [[arXiv version](https://arxiv.org/abs/2007.14598)]
    - Recipe: `egs/pstn`
- Tencent
    - Dataset download link: https://www.dropbox.com/s/ocmn78uh2lu5iwg/TencentCorups.zip?dl=0
    - Paper link: [[arXiv version](https://arxiv.org/abs/2203.16032)]
    - Recipe: `egs/tencent`
- SingMOS
    - Dataset download link: https://drive.google.com/file/d/1DtzZhk3M_jsxUxirPcFRoBhq-dsinOWN/view?usp=drive_link
    - Paper link: [[arXiv version](https://arxiv.org/abs/2406.10911)]
    - Recipe: `egs/singmos`

## Testing sets in MOS-Bench

- BVCC test
    - See descriptions in the [Training datasets in MOS-Bench](#training-datasets-in-mos-bench) section.
    - Benchmark script: `BENCHMARKS/run_bvcc_test.sh`
- BC19 test (VMC'22 OOD track)
    - Dataset download link: https://zenodo.org/records/6572573
    - Paper link: [[VoiceMOS Challenge 2022](https://arxiv.org/abs/2203.11389)]
    - Benchmark script: `BENCHMARKS/run_bc19_test.sh`
- SOMOS test
    - See descriptions in the [Training datasets in MOS-Bench](#training-datasets-in-mos-bench) section.
    - Benchmark script: `BENCHMARKS/run_somos_test.sh`
- SingMOS test (VMC'24 track 2)
    - See descriptions in the [Training datasets in MOS-Bench](#training-datasets-in-mos-bench) section.
    - Benchmark script: `BENCHMARKS/run_singmos_test.sh`
- NISQA FOR/P501/LIVETALK
    - See descriptions in the [Training datasets in MOS-Bench](#training-datasets-in-mos-bench) section.
    - Benchmark script: `BENCHMARKS/run_nisqa_test.sh`
- TMHINT-QI test
    - See descriptions in the [Training datasets in MOS-Bench](#training-datasets-in-mos-bench) section.
    - Benchmark script: `BENCHMARKS/run_tmhint_qi_test.sh`
- VMC'23 track 1a/1b/2/3 (BC2023, SVCC2023, TMHINTQI-(S))
    - Paper link: [[arXiv version](https://arxiv.org/abs/2310.02640)]
    - Benchmark script: `BENCHMARK/run_vmc23_test.sh`
