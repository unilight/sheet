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