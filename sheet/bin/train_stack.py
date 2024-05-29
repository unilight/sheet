#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2024 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

"""Train meta-model for stacking ."""

import argparse
import logging
import os
import pickle
import time

import numpy as np
import torch
import yaml

import sheet
import sheet.datasets
import sheet.models


def main():
    """Run inference process."""
    parser = argparse.ArgumentParser(
        description=(
            "Inference with trained model " "(See detail in bin/inference.py)."
        )
    )
    parser.add_argument(
        "--csv-path",
        required=True,
        type=str,
        help=("csv file path to train stacking model."),
    )
    parser.add_argument(
        "--expdir",
        type=str,
        required=True,
        help="directory to save model.",
    )
    parser.add_argument(
        "--meta-model-config",
        required=True,
        type=str,
        help=("yaml format configuration file for the meta model. "),
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

    # load original config
    with open(os.path.join(args.expdir, "config.yml")) as f:
        config = yaml.load(f, Loader=yaml.Loader)

    # load meta model config
    with open(args.meta_model_config) as f:
        meta_model_config = yaml.load(f, Loader=yaml.Loader)

    config.update(meta_model_config)
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
        wav_only=True,
        allow_cache=False,
    )
    logging.info(f"Number of samples to train meta model = {len(dataset)}.")

    # setup device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # get model
    model_class = getattr(sheet.models, config["model_type"])
    model = model_class(
        config["model_input"],
        num_listeners=config.get("num_listeners", None),
        **config["model_params"],
    )

    # run inference on all models
    checkpoint_paths = sorted(
        [
            os.path.join(args.expdir, p)
            for p in os.listdir(args.expdir)
            if os.path.isfile(os.path.join(args.expdir, p)) and p.endswith("steps.pkl")
        ]
    )
    xs = np.empty((len(dataset), len(checkpoint_paths)))
    for i, checkpoint_path in enumerate(checkpoint_paths):
        # load model
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu")["model"])
        logging.info(f"Loaded model parameters from {checkpoint_path}.")
        model = model.eval().to(device)

        # start inference
        start_time = time.time()
        logging.info("Running inference...")
        with torch.no_grad():
            for j, batch in enumerate(dataset):
                # set up model input
                model_input = batch[config["model_input"]].unsqueeze(0).to(device)
                model_input_lengths = model_input.new_tensor(
                    [model_input.size(1)]
                ).long()

                # model forward
                if config["inference_mode"] == "mean_listener":
                    outputs = model.mean_listener_inference(
                        model_input, model_input_lengths
                    )
                elif config["inference_mode"] == "mean_net":
                    outputs = model.mean_net_inference(model_input, model_input_lengths)
                else:
                    raise NotImplementedError

                # store results
                pred_score = outputs["scores"].cpu().detach().numpy()[0]
                xs[j][i] = pred_score

        total_inference_time = time.time() - start_time
        logging.info("Total inference time = {} secs.".format(total_inference_time))
        logging.info(
            "Average inference speed = {:.3f} sec / sample.".format(
                total_inference_time / len(dataset)
            )
        )

    ys = np.array([batch["avg_score"] for batch in dataset])

    # define meta model
    if config["meta_model_type"] == "Ridge":
        from sklearn.linear_model import Ridge

        meta_model = Ridge(**config["meta_model_params"])
    else:
        raise NotImplementedError

    # train meta model
    start_time = time.time()
    logging.info("Start training meta model...")
    meta_model.fit(xs, ys)
    total_train_time = time.time() - start_time
    logging.info("Total training time = {} secs.".format(total_train_time))

    # save
    with open(os.path.join(args.expdir, "meta_model.pkl"), "wb") as f:
        pickle.dump(meta_model, f)

    with open(os.path.join(args.expdir, "config.yml"), "w") as f:
        yaml.dump(config, f, Dumper=yaml.Dumper)


if __name__ == "__main__":
    main()
