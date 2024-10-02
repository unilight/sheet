# Zero-shot evaluation on benchmarks

**NOTE**: Please do NOT run these recipes in this folder.

## Usage

Let's say you want to benchmark on `vmc23` (which stands for the VoiceMOS Challenge 2023).

1. You need to have a trained model in anothe recipe (ex., `egs/bvcc`).

2. Then, **IN THAT FOLDER**, execute the following:
```
utils/BENCHMARKS/run_vmc23_test.sh --conf XXX.yaml --checkpoint YYY.ckpt
```

## Recipe structure

All the scripts in this folder share the following stage structure:

- Stage -1: Dataset download. Please modify the `db_root` variable in each script to specify where to download the dataset.
- Stage 0: Dataset preparation and csv file generation. They will be stored in `../<benchmark>/data`. (Ex. `../vmc23/data`).
- Stage 1: Inference.