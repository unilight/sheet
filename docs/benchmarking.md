# Benchmarking guide

If you don't want to train SSQA models but just want to get the test sets, you are in the right place.
The design is that **you don't need to run the installation, which can be time-consuming.**

## Data preparation

First, please set the following two variables in `egs/BENCHMARKS/get_all_bencmarks_installation_free.sh`:
- `db_root`: the root directory to save the raw datsets, including the wav files.
- `datadir`: the directory to save the `csv` files.

Then, executing the commands below.

```bash
cd egs/BENCHMARKS

# data download.
# for certain datasets (ex., BVCC or BC19), folow the instructions to finish downloading.
./utils/BENCHMARKS/get_all_bencmarks_installation_free.sh --stage -1 --stop_stage -1

# data preparation. this step generates the csv files.
./utils/BENCHMARKS/get_all_bencmarks_installation_free.sh --stage 0 --stop_stage 0
```

After the data preparation stage is done, you should get the .csv files in `datadir`. Each csv file contains the following columns:

- `wav_path`: **Absolute** paths to the wav files.
- `system_id`: System ID. This is usually for synthetic datasets. For some datasets, this is set to a dump ID.
- `sample_id`: Sample ID. Should be unique within one single dataset.
- `avg_score`: The ground truth MOS of the sample.

## Metric calculation

You may then use your MOS predictor to run inference over the samples in the test set csv files. We suggest you to overwrite (or resave) the csv file by adding a ``answer`` column. Then, you can use the following command to calculate the metrics. Here we provide an example results csv file in [assets/example_results.csv](../assets/example_results.csv), which is based on BVCC test.

```python
python utils/calculate_metrics.py --csv assets/example_results.csv
```

The result will be:

```
[UTT][ MSE = 0.271 | LCC = 0.867 | SRCC = 0.870 | KTAU = 0.693 ] [SYS][ MSE = 0.123 | LCC = 0.9299 | SRCC = 0.9302  | KTAU = 0.777 ]
```