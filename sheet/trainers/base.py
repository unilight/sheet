#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2024 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

import logging
import os
import time
from collections import defaultdict

import torch
from tensorboardX import SummaryWriter
from tqdm import tqdm

from sheet.utils.model_io import freeze_modules


class Trainer(object):
    """Customized trainer module."""

    def __init__(
        self,
        steps,
        epochs,
        data_loader,
        sampler,
        model,
        criterion,
        optimizer,
        scheduler,
        config,
        device=torch.device("cpu"),
    ):
        """Initialize trainer.

        Args:
            steps (int): Initial global steps.
            epochs (int): Initial global epochs.
            data_loader (dict): Dict of data loaders. It must contrain "train" and "dev" loaders.
            model (dict): Dict of models. It must contrain "generator" and "discriminator" models.
            criterion (dict): Dict of criterions. It must contrain "stft" and "mse" criterions.
            optimizer (dict): Dict of optimizers. It must contrain "generator" and "discriminator" optimizers.
            scheduler (dict): Dict of schedulers. It must contrain "generator" and "discriminator" schedulers.
            config (dict): Config dict loaded from yaml format configuration file.
            device (torch.deive): Pytorch device instance.

        """
        self.steps = steps
        self.epochs = epochs
        self.data_loader = data_loader
        self.sampler = sampler
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device
        self.writer = SummaryWriter(config["outdir"])
        self.finish_train = False

        self.total_train_loss = defaultdict(float)
        self.total_eval_loss = defaultdict(float)
        self.reset_eval_results()

        self.gradient_accumulate_steps = self.config.get("gradient_accumulate_steps", 1)

        self.reporter = list()  # each element is [steps: int, results: dict]
        self.original_patience = self.config.get("patience", None)
        self.current_patience = self.config.get("patience", None)

    def run(self):
        """Run training."""
        self.backward_steps = 0
        self.all_loss = 0.0
        self.tqdm = tqdm(
            initial=self.steps, total=self.config["train_max_steps"], desc="[train]"
        )
        while True:
            # train one epoch
            self._train_epoch()

            # check whether training is finished
            if self.finish_train:
                break

        self.tqdm.close()
        logging.info("Finished training.")

    def save_checkpoint(self, checkpoint_path):
        """Save checkpoint.

        Args:
            checkpoint_path (str): Checkpoint path to be saved.

        """
        state_dict = {
            "optimizer": self.optimizer.state_dict(),
            "steps": self.steps,
            "epochs": self.epochs,
        }
        if self.scheduler is not None:
            state_dict["scheduler"] = self.scheduler.state_dict()

        if self.config["distributed"]:
            state_dict["model"] = self.model.module.state_dict()
        else:
            state_dict["model"] = self.model.state_dict()

        if not os.path.exists(os.path.dirname(checkpoint_path)):
            os.makedirs(os.path.dirname(checkpoint_path))
        torch.save(state_dict, checkpoint_path)

    def load_checkpoint(self, checkpoint_path, load_only_params=False):
        """Load checkpoint.

        Args:
            checkpoint_path (str): Checkpoint path to be loaded.
            load_only_params (bool): Whether to load only model parameters.

        """
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        if self.config["distributed"]:
            self.model.module.load_state_dict(state_dict["model"])
        else:
            self.model.load_state_dict(state_dict["model"])
        if not load_only_params:
            self.steps = state_dict["steps"]
            self.epochs = state_dict["epochs"]
            self.optimizer.load_state_dict(state_dict["optimizer"])
            if self.scheduler is not None:
                self.scheduler.load_state_dict(state_dict["scheduler"])

    def _train_step(self, batch):
        """Train model one step."""
        pass

    def _train_epoch(self):
        """Train model one epoch."""
        for train_steps_per_epoch, batch in enumerate(self.data_loader["train"], 1):
            # train one step
            self._train_step(batch)

            # manually update tqdm
            if self.steps > 1 and self.steps % 10 == 0:
                self.tqdm.update(10)

            if self.backward_steps % self.gradient_accumulate_steps > 0:
                continue

            # check interval
            if self.config["rank"] == 0:
                self._check_log_interval()
                self._check_eval_and_save_interval()

            # check whether training is finished
            if self.finish_train:
                return

        # update
        self.epochs += 1
        self.train_steps_per_epoch = train_steps_per_epoch
        logging.info(
            f"(Steps: {self.steps}) Finished {self.epochs} epoch training "
            f"({self.train_steps_per_epoch} steps per epoch)."
        )

        # needed for shuffle in distributed training
        if self.config["distributed"]:
            self.sampler["train"].set_epoch(self.epochs)

    @torch.no_grad()
    def _eval_step(self, batch):
        """Evaluate model one step."""
        pass

    def _eval(self):
        """Evaluate model with dev set."""
        logging.info(f"(Steps: {self.steps}) Start evaluation.")
        # change mode
        self.model.eval()
        start_time = time.time()

        # loop through dev set
        for eval_steps_per_epoch, batch in enumerate(self.data_loader["dev"], 1):
            self._eval_step(batch)

        logging.info(
            f"(Steps: {self.steps}) Finished evaluation "
            f"({time.time() - start_time} secs)."
        )

    @torch.no_grad()
    def _log_metrics_and_save_figures(self):
        """Log metrics and save figures."""
        pass

    def _write_to_tensorboard(self, loss):
        """Write to tensorboard."""
        for key, value in loss.items():
            self.writer.add_scalar(key, value, self.steps)

    def _check_eval_and_save_interval(self):
        if self.steps % self.config["eval_and_save_interval_steps"] == 0:
            # run evaluation on dev set
            self._eval()

            # get metrics and save figures
            self._log_metrics_and_save_figures()

            # get best n steps
            best_n_steps = self.get_and_show_best_n_models()

            # save current if in best n
            if self.steps in best_n_steps:
                current_checkpoint_path = os.path.join(
                    self.config["outdir"], f"checkpoint-{self.steps}steps.pkl"
                )
                self.save_checkpoint(current_checkpoint_path)
                logging.info(
                    f"Saved checkpoint @ {self.steps} steps because it is in best {self.config['keep_nbest_models']}."
                )

                # retstore patience
                if self.original_patience is not None:
                    self.current_patience = self.original_patience
                    logging.info(f"Restoring patience to {self.original_patience}.")
            else:
                # minus patience
                if self.current_patience is not None:
                    self.current_patience -= 1
                    logging.info(f"Reducing patience to {self.current_patience}.")

            # if current is best, link to best
            if self.steps == best_n_steps[0]:
                best_checkpoint_path = os.path.join(
                    self.config["outdir"], f"checkpoint-best.pkl"
                )
                if os.path.islink(best_checkpoint_path) or os.path.exists(
                    best_checkpoint_path
                ):
                    os.remove(best_checkpoint_path)
                os.symlink(current_checkpoint_path, best_checkpoint_path)
                logging.info(f"Updated best checkpoint to {self.steps} steps.")

            # delete those not in best n
            existing_checkpoint_paths = [
                fname
                for fname in os.listdir(self.config["outdir"])
                if os.path.isfile(os.path.join(self.config["outdir"], fname))
                and fname.endswith("steps.pkl")
            ]
            for checkpoint_path in existing_checkpoint_paths:
                steps = int(
                    checkpoint_path.replace("steps.pkl", "").replace("checkpoint-", "")
                )
                if steps not in best_n_steps:
                    os.remove(os.path.join(self.config["outdir"], checkpoint_path))
                    logging.info(f"Deleting checkpoint @ {steps} steps.")

            # reset
            self.reset_eval_results()

            # restore mode
            self.model.train()

    def _check_log_interval(self):
        if self.steps % self.config["log_interval_steps"] == 0:
            for key in self.total_train_loss.keys():
                self.total_train_loss[key] /= self.config["log_interval_steps"]
                logging.info(
                    f"(Steps: {self.steps}) {key} = {self.total_train_loss[key]:.4f}."
                )
            self._write_to_tensorboard(self.total_train_loss)

            # reset
            self.total_train_loss = defaultdict(float)

    def _check_train_finish(self):
        if self.steps >= self.config["train_max_steps"]:
            self.finish_train = True

        if self.current_patience is not None and self.current_patience <= 0:
            self.finish_train = True

    def freeze_modules(self, modules):
        freeze_modules(self.model, modules)

    def reset_eval_results(self):
        self.eval_results = defaultdict(list)
        self.eval_sys_results = defaultdict(lambda: defaultdict(list))

    def get_and_show_best_n_models(self):
        # sort according to key
        best_n = sorted(
            self.reporter,
            key=lambda x: x[1][self.config["best_model_criterion"]["key"]],
        )
        if (
            self.config["best_model_criterion"]["order"] == "highest"
        ):  # reverse if highest
            best_n.reverse()

        # log the results
        logging.info(
            f"Best {self.config['keep_nbest_models']} models at step {self.steps}:"
        )
        log_string = "; ".join(
            f"{i+1}. {steps} steps: {self.config['best_model_criterion']['key']}={results[self.config['best_model_criterion']['key']]:.4f}"
            for i, (steps, results) in enumerate(
                best_n[: self.config["keep_nbest_models"]]
            )
        )
        logging.info(log_string)

        # only return the steps
        return [steps for steps, _ in best_n[: self.config["keep_nbest_models"]]]
