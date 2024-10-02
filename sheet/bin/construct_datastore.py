#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2024 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

"""Construct datastore ."""

import argparse
import h5py
import logging
import os

import numpy as np
import sheet
import sheet.datasets
import sheet.models
import torch
import yaml
from s3prl.nn import S3PRLUpstream
from tqdm import tqdm


def main():
    """Construct datastore."""
    parser = argparse.ArgumentParser(
        description=(
            "Construct datastore with ssl_model in trained model " "(See detail in bin/construct_datastore.py)."
        )
    )
    parser.add_argument(
        "--csv-path",
        required=True,
        type=str,
        help=("csv file path to construct datastore."),
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="out path to save datastore.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="checkpoint file to be loaded.",
    )
    parser.add_argument(
        "--config",
        default=None,
        type=str,
        help=(
            "yaml format configuration file. if not explicitly provided, "
            "it will be searched in the checkpoint directory. (default=None)"
        ),
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="logging level. higher is more logging. (default=1)",
    )
    args = parser.parse_args()

    # set logger
    if args.verbose > 1:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    elif args.verbose > 0:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    else:
        logging.basicConfig(
            level=logging.WARN,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
        logging.warning("Skip DEBUG/INFO messages")

    # check directory existence
    # if not os.path.exists(args.outdir):
        # os.makedirs(args.outdir)

    # load config
    if args.config is None:
        dirname = os.path.dirname(args.checkpoint)
        args.config = os.path.join(dirname, "config.yml")
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.Loader)

    args_dict = vars(args)

    config.update(args_dict)
    for key, value in config.items():
        logging.info(f"{key} = {value}")

    # get dataset
    dataset_class = getattr(
        sheet.datasets,
        config.get("dataset_type", "NonIntrusiveDataset"),
    )
    dataset = dataset_class(
        csv_path=args.csv_path,
        target_sample_rate=config["sampling_rate"],
        model_input=config["model_input"],
        use_phoneme=config.get("use_phoneme", False),
        symbols=config.get("symbols", None),
        wav_only=True,
        allow_cache=False,
    )
    logging.info(f"Number of samples = {len(dataset)}.")

    # setup device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # get ssl model
    s3prl_name = config["model_params"]["s3prl_name"]
    ssl_model = S3PRLUpstream(s3prl_name)

    # load pre-trained model
    pt_ckpt = torch.load(os.readlink(args.checkpoint), map_location="cpu")["model"]
    state_dict = {
        k.replace("ssl_model.", ""): v
        for k, v in pt_ckpt.items()
        if k.startswith("ssl_model")
    }
    ssl_model.load_state_dict(state_dict)
    logging.info(f"Loaded model parameters from {args.checkpoint}.")
    ssl_model = ssl_model.eval().to(device)

    # start inference
    if os.path.exists(args.out):
        hdf5_file = h5py.File(args.out, "r+")
    else:
        hdf5_file = h5py.File(args.out, "w")

    with torch.no_grad(), tqdm(dataset, desc="[inference]") as pbar:
        for batch in pbar:
            # set up model input
            model_input = batch[config["model_input"]].unsqueeze(0).to(device)
            model_lengths = model_input.new_tensor([model_input.size(1)]).long()

            all_encoder_outputs, _ = ssl_model(model_input, model_lengths)
            embed = torch.mean(all_encoder_outputs[config["model_params"]["ssl_model_layer_idx"]].squeeze(0), dim=0).detach().cpu().numpy()

            system_id = batch["system_id"]
            sample_id = batch["sample_id"]
            hdf5_path = system_id + "_" + sample_id
            score = batch["avg_score"]

            hdf5_file.create_dataset("embeds/" + hdf5_path, data=embed)
            hdf5_file.create_dataset("scores/" + hdf5_path, data=score)

    hdf5_file.flush()
    hdf5_file.close()

if __name__ == "__main__":
    main()
