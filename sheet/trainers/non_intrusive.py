#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2024 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

import logging
import os
import time

# set to avoid matplotlib error in CLI environment
import matplotlib
import numpy as np
import soundfile as sf
import torch
from sheet.evaluation.metrics import calculate

from sheet.utils.model_io import (
    filter_modules,
    get_partial_state_dict,
    transfer_verification,
    print_new_keys,
)
from sheet.evaluation.plot import (
    plot_sys_level_scatter,
    plot_utt_level_hist,
    plot_utt_level_scatter,
)
from sheet.trainers.base import Trainer

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class NonIntrusiveEstimatorTrainer(Trainer):
    """Customized trainer module for non-intrusive estimator."""

    def load_trained_modules(self, checkpoint_path, init_mods):
        if self.config["distributed"]:
            main_state_dict = self.model.module.state_dict()
        else:
            main_state_dict = self.model.state_dict()

        if os.path.isfile(checkpoint_path):
            model_state_dict = torch.load(checkpoint_path, map_location="cpu")["model"]

            # first make sure that all modules in `init_mods` are in `checkpoint_path`
            modules = filter_modules(model_state_dict, init_mods)

            # then, actually get the partial state_dict
            partial_state_dict = get_partial_state_dict(model_state_dict, modules)

            if partial_state_dict:
                if transfer_verification(main_state_dict, partial_state_dict, modules):
                    print_new_keys(partial_state_dict, modules, checkpoint_path)
                    main_state_dict.update(partial_state_dict)
        else:
            logging.error(f"Specified model was not found: {checkpoint_path}")
            exit(1)

        if self.config["distributed"]:
            self.model.module.load_state_dict(main_state_dict)
        else:
            self.model.load_state_dict(main_state_dict)

    def _train_step(self, batch):
        """Train model one step."""

        # set inputs
        gen_loss = 0.0
        inputs = {
            self.config["model_input"]: batch[self.config["model_input"]].to(
                self.device
            ),
            self.config["model_input"]
            + "_lengths": batch[self.config["model_input"] + "_lengths"].to(
                self.device
            ),
        }
        if "listener_idxs" in batch:
            inputs["listener_idxs"] = batch["listener_idxs"].to(self.device)
        if "domain_idxs" in batch:
            inputs["domain_idxs"] = batch["domain_idxs"].to(self.device)
        if "phoneme_idxs" in batch:
            inputs["phoneme_idxs"] = batch["phoneme_idxs"].to(self.device)
            inputs["phoneme_lengths"] = batch["phoneme_lengths"]
        if "reference_idxs" in batch:
            inputs["reference_idxs"] = batch["reference_idxs"].to(self.device)
            inputs["reference_lengths"] = batch["reference_lengths"]

        # model forward
        outputs = self.model(inputs)

        # get frame lengths if exist
        if "frame_lengths" in outputs:
            output_frame_lengths = outputs["frame_lengths"]
        else:
            output_frame_lengths = None

        # get ground truth scores
        gt_scores = batch["scores"].to(self.device)
        gt_avg_scores = batch["avg_scores"].to(self.device)

        # mean loss
        if "mean_score_criterions" in self.criterion:
            for criterion_dict in self.criterion["mean_score_criterions"]:
                # always pass the following arguments
                loss = criterion_dict["criterion"](
                    outputs["mean_scores"],
                    gt_avg_scores,
                    self.device,
                    lens = output_frame_lengths,
                )
                gen_loss += loss * criterion_dict["weight"]
                self.total_train_loss[
                    "train/mean_" + criterion_dict["type"]
                ] += loss.item()

        # listener loss
        if "listener_score_criterions" in self.criterion:
            for criterion_dict in self.criterion["listener_score_criterions"]:
                # always pass the following arguments
                loss = criterion_dict["criterion"](
                    outputs["ld_scores"],
                    gt_scores,
                    self.device,
                    lens = output_frame_lengths,
                )
                gen_loss += loss * criterion_dict["weight"]
                self.total_train_loss[
                    "train/listener_" + criterion_dict["type"]
                ] += loss.item()

        self.total_train_loss["train/loss"] += gen_loss.item()

        # update model
        self.optimizer.zero_grad()
        gen_loss.backward()
        if self.config["grad_norm"] > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config["grad_norm"],
            )
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        # update counts
        self.steps += 1
        self._check_train_finish()

    @torch.no_grad()
    def _eval_step(self, batch):
        """Evaluate model one step."""

        # set up model input
        inputs = {
            self.config["model_input"]: batch[self.config["model_input"]].to(
                self.device
            ),
            self.config["model_input"]
            + "_lengths": batch[self.config["model_input"] + "_lengths"].to(
                self.device
            ),
        }
        if "domain_idxs" in batch:
            inputs["domain_idxs"] = batch["domain_idxs"].to(self.device)
        if "phoneme_idxs" in batch:
            inputs["phoneme_idxs"] = batch["phoneme_idxs"].to(self.device)
            inputs["phoneme_lengths"] = batch["phoneme_lengths"]
        if "reference_idxs" in batch:
            inputs["reference_idxs"] = batch["reference_idxs"].to(self.device)
            inputs["reference_lengths"] = batch["reference_lengths"]

        # model forward
        if self.config["inference_mode"] == "mean_listener":
            outputs = self.model.mean_listener_inference(inputs)
        elif self.config["inference_mode"] == "mean_net":
            outputs = self.model.mean_net_inference(inputs)

        # construct the eval_results dict
        pred_mean_scores = outputs["scores"].cpu().detach().numpy()
        true_mean_scores = batch["avg_scores"].numpy()
        self.eval_results["pred_mean_scores"].extend(pred_mean_scores.tolist())
        self.eval_results["true_mean_scores"].extend(true_mean_scores.tolist())
        sys_names = batch["system_ids"]
        for j, sys_name in enumerate(sys_names):
            self.eval_sys_results["pred_mean_scores"][sys_name].append(
                pred_mean_scores[j]
            )
            self.eval_sys_results["true_mean_scores"][sys_name].append(
                true_mean_scores[j]
            )

    @torch.no_grad()
    def _log_metrics_and_save_figures(self):
        """Log metrics and save figures."""

        self.eval_results["true_mean_scores"] = np.array(
            self.eval_results["true_mean_scores"]
        )
        self.eval_results["pred_mean_scores"] = np.array(
            self.eval_results["pred_mean_scores"]
        )
        self.eval_sys_results["true_mean_scores"] = np.array(
            [
                np.mean(scores)
                for scores in self.eval_sys_results["true_mean_scores"].values()
            ]
        )
        self.eval_sys_results["pred_mean_scores"] = np.array(
            [
                np.mean(scores)
                for scores in self.eval_sys_results["pred_mean_scores"].values()
            ]
        )

        # calculate metrics
        results = calculate(
            self.eval_results["true_mean_scores"],
            self.eval_results["pred_mean_scores"],
            self.eval_sys_results["true_mean_scores"],
            self.eval_sys_results["pred_mean_scores"],
        )

        # log metrics
        logging.info(
            f'[{self.steps} steps][UTT][ MSE = {results["utt_MSE"]:.3f} | LCC = {results["utt_LCC"]:.3f} | SRCC = {results["utt_SRCC"]:.3f} ] [SYS][ MSE = {results["sys_MSE"]:.3f} | LCC = {results["sys_LCC"]:.4f} | SRCC = {results["sys_SRCC"]:.4f} ]\n'
        )

        # register metrics to reporter
        self.reporter.append([self.steps, results])

        # check directory
        dirname = os.path.join(
            self.config["outdir"], f"intermediate_results/{self.steps}steps"
        )
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        # plot
        plot_utt_level_hist(
            self.eval_results["true_mean_scores"],
            self.eval_results["pred_mean_scores"],
            os.path.join(dirname, "distribution.png"),
        )
        plot_utt_level_scatter(
            self.eval_results["true_mean_scores"],
            self.eval_results["pred_mean_scores"],
            os.path.join(dirname, "utt_scatter_plot.png"),
            results["utt_LCC"],
            results["utt_SRCC"],
            results["utt_MSE"],
            results["utt_KTAU"],
        )
        plot_sys_level_scatter(
            self.eval_sys_results["true_mean_scores"],
            self.eval_sys_results["pred_mean_scores"],
            os.path.join(dirname, "sys_scatter_plot.png"),
            results["sys_LCC"],
            results["sys_SRCC"],
            results["sys_MSE"],
            results["sys_KTAU"],
        )
