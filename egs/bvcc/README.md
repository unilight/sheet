# BVCC

## Supported models

- LDNet (`conf/ldnet-ml.yaml`)
- SSL-MOS (`conf/ssl-mos-wav2vec2.yaml`)
- UTMOS Strong (`conf/utmos-strong.yaml`)

## Usage

In this example we use the SSL-MOS model (with the config file `conf/ssl-mos-wav2vec2.yaml`). 

### Main track

#### Data download

There is not an one-step automatic download script due to the data policy of the Blizzard Challenge. Please follow the instructions and put the data somewhere, and properly set the `db_root` variable in `run.sh`.

#### Preparation

```
./run.sh --stage 0 --stop_stage 0
```

The processed data files are in `data`.

#### Training

```
./run.sh --stage 2 --stop_stage 2
```

The model checkpoints can be found in `exp/ssl-mos-wav2vec2-1337`. The `1337` is the random seed. Also, you can check the `exp/ssl-mos-wav2vec2-1337/intermediate_results` to see some plots to monitor the training process.

#### Inference

```
./run.sh --stage 4 --stop_stage 4
```

By default, the `checkpoint-best.pkl` is used, which is a symbolic link that points to the best performing model checkpoint (depending on the `best_model_criterion` field in the config file.) You can find the inference log and the plots in `exp/ssl-mos-wav2vec2-1337/results/<set_name>/`.


## Notes

- By default, the phonemes and references provided by the UTMOS authors are always directly downloaded. Currently we have not supported transcribing datasets other than UTMOS.

## Results

- `conf/ssl-mos-wav2vec2.yaml`, seed=1337, 26400 steps

| dataset           | Utt MSE | Utt LCC | Utt SRCC | Sys MSE | Sys LCC | Sys SRCC |
| :-----------------|-------: |-------: |--------: |-------: |-------: |--------: |
| bvcc_dev          |   0.230 |   0.875 |    0.875 |   0.111 |   0.952 |    0.951 |
| bvcc_test         |   0.209 |   0.879 |    0.879 |   0.121 |   0.915 |    0.914 |
| nisqa_FOR         |   2.011 |  -0.026 |   -0.018 |   1.931 |  -0.007 |    0.048 |
| nisqa_LIVETALK    |   2.933 |   0.043 |    0.116 |   2.706 |   0.030 |    0.093 |
| nisqa_P501        |   2.669 |  -0.010 |    0.001 |   2.563 |  -0.001 |    0.073 |
| singmos_dev       |   4.896 |   0.382 |    0.396 |   4.371 |   0.760 |    0.718 |
| singmos_test      |   4.635 |   0.323 |    0.358 |   4.217 |   0.493 |    0.499 |
| somos_test        |   0.655 |   0.450 |    0.448 |   0.424 |   0.677 |    0.706 |
| tmhintqi_dev      |   3.730 |   0.303 |    0.293 |   2.843 |   0.375 |    0.396 |
| tmhintqi_test     |   3.033 |   0.517 |    0.416 |   2.605 |   0.374 |    0.514 |
| vmc23_track1_test |   3.062 |   0.138 |    0.142 |   2.947 |   0.134 |    0.194 |
| vmc23_track2_test |   3.393 |   0.231 |    0.273 |   3.144 |   0.600 |    0.603 |
| vmc23_track3_test |   2.426 |   0.487 |    0.450 |   2.293 |   0.775 |    0.586 |

- `conf/utmos-strong-wo-phoneme.yaml`, seed=1337, 11700 steps

| dataset           | Utt MSE | Utt LCC | Utt SRCC | Sys MSE | Sys LCC | Sys SRCC |
| :-----------------|-------: |-------: |--------: |-------: |-------: |--------: |
| bvcc_dev          |   0.369 |   0.878 |    0.893 |   0.291 |   0.936 |    0.956 |
| bvcc_test         |   0.303 |   0.872 |    0.876 |   0.163 |   0.923 |    0.921 |
| nisqa_FOR         |  17.328 |  -0.207 |   -0.165 |  17.250 |  -0.210 |   -0.170 |
| nisqa_LIVETALK    |  20.413 |  -0.026 |   -0.019 |  20.184 |  -0.012 |   -0.073 |
| nisqa_P501        |  19.147 |   0.023 |    0.052 |  19.052 |   0.032 |    0.056 |
| singmos_dev       |  24.200 |   0.390 |    0.393 |  23.575 |   0.680 |    0.645 |
| singmos_test      |  23.633 |   0.345 |    0.342 |  23.053 |   0.426 |    0.394 |
| somos_dev         |   7.358 |   0.444 |    0.430 |   7.232 |   0.612 |    0.657 |
| somos_test        |   7.361 |   0.438 |    0.428 |   7.100 |   0.638 |    0.666 |
| tmhintqi_dev      |  21.884 |   0.379 |    0.313 |  21.188 |   0.521 |    0.356 |
| tmhintqi_test     |  21.067 |   0.606 |    0.441 |  20.805 |   0.508 |    0.423 |
| vmc23_track1_test |  19.957 |   0.106 |    0.139 |  20.029 |   0.085 |    0.129 |
| vmc23_track2_test |  22.280 |   0.150 |    0.239 |  22.194 |   0.452 |    0.423 |
| vmc23_track3_test |  18.593 |   0.605 |    0.388 |  19.123 |   0.747 |    0.469 |