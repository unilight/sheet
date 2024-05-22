#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2024 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

"""Inference ."""

import argparse
import logging
import os
import time
from collections import defaultdict

import numpy as np
import soundfile as sf
import torch
import yaml
from tqdm import tqdm

import sheet
import sheet.datasets
import sheet.models
from sheet.evaluation.metrics import calculate
from sheet.evaluation.plot import (
    plot_sys_level_scatter,
    plot_utt_level_hist,
    plot_utt_level_scatter,
)


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
        help=("csv file path to do inference."),
    )
    parser.add_argument(
        "--outdir",
        type=str,
        required=True,
        help="directory to save generated speech.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
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
    parser.add_argument(
        "--inference-mode",
        type=str,
        help="inference mode. if not specified, use the default setting in config",
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
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # load config
    if args.config is None:
        dirname = os.path.dirname(args.checkpoint)
        args.config = os.path.join(dirname, "config.yml")
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    
    args_dict = vars(args)
    # do not override if inference mode not specified
    if args_dict["inference_mode"] is None:
        del args_dict["inference_mode"]

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
        model_input=config["model_input"],
        wav_only=True,
        allow_cache=False,
    )
    logging.info(f"Number of inference samples = {len(dataset)}.")

    # setup device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # get model and load parameters
    model_class = getattr(sheet.models, config["model_type"])
    model = model_class(
        config["model_input"],
        num_listeners=config.get("num_listeners", None),
        **config["model_params"],
    )
    model.load_state_dict(torch.load(os.readlink(args.checkpoint), map_location="cpu")["model"])
    model = model.eval().to(device)
    logging.info(f"Loaded model parameters from {args.checkpoint}.")

    # set placeholders
    eval_results = defaultdict(list)
    eval_sys_results = defaultdict(lambda: defaultdict(list))

    # start inference
    start_time = time.time()
    with torch.no_grad(), tqdm(dataset, desc="[inference]") as pbar:
        for batch in pbar:
            # set up model input
            model_input = batch[config["model_input"]].unsqueeze(0).to(device)
            model_input_lengths = model_input.new_tensor([model_input.size(1)]).long()

            # model forward
            if config["inference_mode"] == "mean_listener":
                outputs = model.mean_listener_inference(model_input, model_input_lengths)
            elif config["inference_mode"] == "mean_net":
                outputs = model.mean_net_inference(model_input, model_input_lengths)
            else:
                raise NotImplementedError

            # store results
            pred_mean_scores = outputs["scores"].cpu().detach().numpy()[0]
            true_mean_scores = batch["avg_score"]
            eval_results["pred_mean_scores"].append(pred_mean_scores)
            eval_results["true_mean_scores"].append(true_mean_scores)
            sys_name = batch["system_id"]
            eval_sys_results["pred_mean_scores"][sys_name].append(pred_mean_scores)
            eval_sys_results["true_mean_scores"][sys_name].append(true_mean_scores)

    total_inference_time = time.time() - start_time
    logging.info("Total inference time = {} secs.".format(total_inference_time))
    logging.info(
        "Average inference speed = {:.3f} sec / sample.".format(
            total_inference_time / len(dataset)
        )
    )

    # calculate metrics
    eval_results["true_mean_scores"] = np.array(eval_results["true_mean_scores"])
    eval_results["pred_mean_scores"] = np.array(eval_results["pred_mean_scores"])
    eval_sys_results["true_mean_scores"] = np.array(
        [np.mean(scores) for scores in eval_sys_results["true_mean_scores"].values()]
    )
    eval_sys_results["pred_mean_scores"] = np.array(
        [np.mean(scores) for scores in eval_sys_results["pred_mean_scores"].values()]
    )

    # calculate metrics
    results = calculate(
        eval_results["true_mean_scores"],
        eval_results["pred_mean_scores"],
        eval_sys_results["true_mean_scores"],
        eval_sys_results["pred_mean_scores"],
    )
    logging.info(
        f'[UTT][ MSE = {results["utt_MSE"]:.3f} | LCC = {results["utt_LCC"]:.3f} | SRCC = {results["utt_SRCC"]:.3f} ] [SYS][ MSE = {results["sys_MSE"]:.3f} | LCC = {results["sys_LCC"]:.4f} | SRCC = {results["sys_SRCC"]:.4f} ]\n'
    )

    # check directory
    dirname = args.outdir
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    # plot
    plot_utt_level_hist(
        eval_results["true_mean_scores"],
        eval_results["pred_mean_scores"],
        os.path.join(dirname, "distribution.png"),
    )
    plot_utt_level_scatter(
        eval_results["true_mean_scores"],
        eval_results["pred_mean_scores"],
        os.path.join(dirname, "utt_scatter_plot.png"),
        results["utt_LCC"],
        results["utt_SRCC"],
        results["utt_MSE"],
        results["utt_KTAU"],
    )
    plot_sys_level_scatter(
        eval_sys_results["true_mean_scores"],
        eval_sys_results["pred_mean_scores"],
        os.path.join(dirname, "sys_scatter_plot.png"),
        results["sys_LCC"],
        results["sys_SRCC"],
        results["sys_MSE"],
        results["sys_KTAU"],
    )


if __name__ == "__main__":
    main()
