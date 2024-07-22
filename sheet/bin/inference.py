#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2024 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

"""Inference ."""

import argparse
import logging
import os
import pickle
from prettytable import PrettyTable, MARKDOWN
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
from sheet.utils.model_io import model_average
from sheet.utils.types import str2bool


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
        help="directory to save generated figures.",
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
    parser.add_argument(
        "--inference-mode",
        type=str,
        help="inference mode. if not specified, use the default setting in config",
    )
    parser.add_argument(
        "--model-averaging",
        type=str2bool,
        default="False",
        help="if true, average all model checkpoints in the exp directory",
    )
    parser.add_argument(
        "--use-stacking",
        type=str2bool,
        default="False",
        help="if true, use the stack model in the exp directory",
    )
    parser.add_argument(
        "--meta-model-checkpoint",
        type=str,
        help="checkpoint file of meta model.",
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

    # get expdir first
    expdir = config["outdir"]

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
    logging.info(f"Number of inference samples = {len(dataset)}.")

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

    # set placeholders
    eval_results = defaultdict(list)
    eval_sys_results = defaultdict(lambda: defaultdict(list))

    # stacking model inference
    if args.use_stacking:
        # load meta model
        with open(args.meta_model_checkpoint, "rb") as f:
            meta_model = pickle.load(f)

        # run inference on all models
        checkpoint_paths = sorted(
            [
                os.path.join(expdir, p)
                for p in os.listdir(expdir)
                if os.path.isfile(os.path.join(expdir, p)) and p.endswith("steps.pkl")
            ]
        )
        xs = np.empty((len(dataset), len(checkpoint_paths)))
        for i, checkpoint_path in enumerate(checkpoint_paths):
            # load model
            model.load_state_dict(
                torch.load(checkpoint_path, map_location="cpu")["model"]
            )
            logging.info(f"Loaded model parameters from {checkpoint_path}.")
            model = model.eval().to(device)

            # start inference
            start_time = time.time()
            logging.info("Running inference...")
            with torch.no_grad():
                for j, batch in enumerate(dataset):
                    # set up model input
                    model_input = batch[config["model_input"]].unsqueeze(0).to(device)
                    model_lengths = model_input.new_tensor(
                            [model_input.size(1)]
                        ).long()
                    inputs = {
                        config["model_input"]: model_input,
                        config["model_input"] + "_lengths": model_lengths
                    }
                    if "phoneme_idxs" in batch:
                        inputs["phoneme_idxs"] = batch["phoneme_idxs"].unsqueeze(0).to(device)
                        inputs["phoneme_lengths"] = batch["phoneme_lengths"]
                    if "reference_idxs" in batch:
                        inputs["reference_idxs"] = batch["reference_idxs"].unsqueeze(0).to(device)
                        inputs["reference_lengths"] = batch["reference_lengths"]

                    # model forward
                    if config["inference_mode"] == "mean_listener":
                        outputs = model.mean_listener_inference(inputs)
                    elif config["inference_mode"] == "mean_net":
                        outputs = model.mean_net_inference(inputs)
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

        # run inference on meta model
        pred_mean_scores = meta_model.predict(xs)

        # rerun dataset to get system level scores
        for i, batch in enumerate(dataset):
            true_mean_scores = batch["avg_score"]
            eval_results["pred_mean_scores"].append(pred_mean_scores[i])
            eval_results["true_mean_scores"].append(true_mean_scores)
            sys_name = batch["system_id"]
            eval_sys_results["pred_mean_scores"][sys_name].append(pred_mean_scores[i])
            eval_sys_results["true_mean_scores"][sys_name].append(true_mean_scores)
    else:
        # load parameter, or take average
        assert (args.checkpoint == "" and args.model_averaging) or (
            args.checkpoint != "" and not args.model_averaging
        )
        if args.checkpoint != "":
            model.load_state_dict(
                torch.load(os.readlink(args.checkpoint), map_location="cpu")["model"]
            )
            logging.info(f"Loaded model parameters from {args.checkpoint}.")
        else:
            model, checkpoint_paths = model_average(model, expdir)
            logging.info(f"Loaded model parameters from: {', '.join(checkpoint_paths)}")
        model = model.eval().to(device)

        # start inference
        start_time = time.time()
        with torch.no_grad(), tqdm(dataset, desc="[inference]") as pbar:
            for batch in pbar:
                # set up model input
                model_input = batch[config["model_input"]].unsqueeze(0).to(device)
                model_lengths = model_input.new_tensor(
                        [model_input.size(1)]
                    ).long()
                inputs = {
                    config["model_input"]: model_input,
                    config["model_input"] + "_lengths": model_lengths
                }
                if "phoneme_idxs" in batch:
                    inputs["phoneme_idxs"] = torch.tensor(batch["phoneme_idxs"], dtype=torch.long).unsqueeze(0).to(device)
                    inputs["phoneme_lengths"] = torch.tensor([len(batch["phoneme_idxs"])], dtype=torch.long)
                if "reference_idxs" in batch:
                    inputs["reference_idxs"] = torch.tensor(batch["reference_idxs"], dtype=torch.long).unsqueeze(0).to(device)
                    inputs["reference_lengths"] = torch.tensor([len(batch["reference_idxs"])], dtype=torch.long)

                # model forward
                if config["inference_mode"] == "mean_listener":
                    outputs = model.mean_listener_inference(inputs)
                elif config["inference_mode"] == "mean_net":
                    outputs = model.mean_net_inference(inputs)
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
    
    table = PrettyTable()
    table.set_style(MARKDOWN)
    table.field_names = ["Utt MSE", "Utt LCC", "Utt SRCC", "Utt KTAU", "Sys MSE", "Sys LCC", "Sys SRCC", "Sys KTAU"]
    table.add_row([
        round(results["utt_MSE"], 3),
        round(results["utt_LCC"], 3),
        round(results["utt_SRCC"], 3),
        round(results["utt_KTAU"], 3),
        round(results["sys_MSE"], 3),
        round(results["sys_LCC"], 3),
        round(results["sys_SRCC"], 3),
        round(results["sys_KTAU"], 3)
    ])
    print(table)

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
