#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2024 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

"""Train model."""

import argparse
import logging
import os
import sys

import humanfriendly
import numpy as np
import sheet
import sheet.collaters
import sheet.datasets
import sheet.losses
import sheet.models
import sheet.trainers
import torch
import yaml
from sheet.schedulers import get_scheduler
from torch.utils.data import DataLoader

# scheduler_classes = dict(warmuplr=WarmupLR)


def main():
    """Run training process."""
    parser = argparse.ArgumentParser(
        description=(
            "Train speech human evaluation estimation model (See detail in bin/train.py)."
        )
    )
    parser.add_argument(
        "--train-csv-path",
        required=True,
        type=str,
        help=("training data csv file path."),
    )
    parser.add_argument(
        "--dev-csv-path",
        required=True,
        type=str,
        help=("training data csv file path."),
    )
    parser.add_argument(
        "--outdir",
        type=str,
        required=True,
        help="directory to save checkpoints.",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="yaml format configuration file.",
    )
    parser.add_argument(
        "--additional-config",
        type=str,
        default=None,
        help="yaml format configuration file (additional; for second-stage pretraining).",
    )
    parser.add_argument(
        "--init-checkpoint",
        default="",
        type=str,
        nargs="?",
        help='checkpoint file path to initialize pretrained params. (default="")',
    )
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        nargs="?",
        help='checkpoint file path to resume training. (default="")',
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="logging level. higher is more logging. (default=1)",
    )
    parser.add_argument(
        "--rank",
        "--local_rank",
        default=0,
        type=int,
        help="rank for distributed training. no need to explictly specify.",
    )
    parser.add_argument("--seed", default=1337, type=int)
    args = parser.parse_args()

    args.distributed = False
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
        # effective when using fixed size inputs
        # see https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936
        torch.backends.cudnn.benchmark = False  # because we have dynamic input size
        torch.cuda.set_device(args.rank)
        # setup for distributed training
        # see example: https://github.com/NVIDIA/apex/tree/master/examples/simple/distributed
        if "WORLD_SIZE" in os.environ:
            args.world_size = int(os.environ["WORLD_SIZE"])
            args.distributed = args.world_size > 1
        if args.distributed:
            torch.distributed.init_process_group(backend="nccl", init_method="env://")

    # suppress logging for distributed training
    if args.rank != 0:
        sys.stdout = open(os.devnull, "w")

    # set logger
    if args.verbose > 1:
        logging.basicConfig(
            level=logging.DEBUG,
            stream=sys.stdout,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    elif args.verbose > 0:
        logging.basicConfig(
            level=logging.INFO,
            stream=sys.stdout,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    else:
        logging.basicConfig(
            level=logging.WARN,
            stream=sys.stdout,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
        logging.warning("Skip DEBUG/INFO messages")

    # Fix seed and make backends deterministic
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    # fix issue of too many opened files
    # https://github.com/pytorch/pytorch/issues/11201
    torch.multiprocessing.set_sharing_strategy("file_system")

    # check directory existence
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # load main config
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config.update(vars(args))

    # load additional config
    if args.additional_config is not None:
        with open(args.additional_config) as f:
            additional_config = yaml.load(f, Loader=yaml.Loader)
        config.update(additional_config)

    # get dataset
    dataset_class = getattr(
        sheet.datasets,
        config.get("dataset_type", "NonIntrusiveDataset"),
    )
    logging.info(f"Loading training set from {args.train_csv_path}.")
    train_dataset = dataset_class(
        csv_path=args.train_csv_path,
        target_sample_rate=config["sampling_rate"],
        model_input=config["model_input"],
        wav_only=config.get("wav_only", False),
        use_phoneme=config.get("use_phoneme", False),
        symbols=config.get("symbols", None),
        use_mean_listener=config["model_params"].get("use_mean_listener", None),
        categorical=config.get("categorical", False),
        categorical_step=config.get("categorical_step", 1.0),
        allow_cache=config["allow_cache"],
        load_wav_cache=config.get("load_wav_cache", False),
    )
    logging.info(f"The number of training files = {len(train_dataset)}.")
    logging.info(f"Loading development set from {args.dev_csv_path}.")
    dev_dataset = dataset_class(
        csv_path=args.dev_csv_path,
        target_sample_rate=config["sampling_rate"],
        model_input=config["model_input"],
        wav_only=True,
        use_phoneme=config.get("use_phoneme", False),
        symbols=config.get("symbols", None),
        allow_cache=False,
        # allow_cache=config["allow_cache"],
        # load_wav_cache=config.get("load_wav_cache", False),
    )
    logging.info(f"The number of development files = {len(dev_dataset)}.")
    dataset = {
        "train": train_dataset,
        "dev": dev_dataset,
    }

    # update number of listeners
    if hasattr(train_dataset, "num_listeners"):
        config["num_listeners"] = train_dataset.num_listeners

    # update number of domains
    if config.get("num_domains", None) is None:
        if hasattr(train_dataset, "num_domains"):
            config["num_domains"] = train_dataset.num_domains

    # get data loader
    collater_class = getattr(
        sheet.collaters,
        config.get("collater_type", "NonIntrusiveCollater"),
    )
    collater = collater_class(config["padding_mode"])
    sampler = {"train": None, "dev": None}
    if args.distributed:
        # setup sampler for distributed training
        from torch.utils.data.distributed import DistributedSampler

        sampler["train"] = DistributedSampler(
            dataset=dataset["train"],
            num_replicas=args.world_size,
            rank=args.rank,
            shuffle=True,
        )
        sampler["dev"] = DistributedSampler(
            dataset=dataset["dev"],
            num_replicas=args.world_size,
            rank=args.rank,
            shuffle=False,
        )
    data_loader = {
        "train": DataLoader(
            dataset=dataset["train"],
            shuffle=False if args.distributed else True,
            collate_fn=collater,
            batch_size=config["train_batch_size"],
            num_workers=config["num_workers"],
            sampler=sampler["train"],
            pin_memory=config["pin_memory"],
        ),
        "dev": DataLoader(
            dataset=dataset["dev"],
            shuffle=False,
            collate_fn=collater,
            batch_size=config["test_batch_size"],
            num_workers=config["num_workers"],
            sampler=sampler["dev"],
            pin_memory=config["pin_memory"],
        ),
    }

    # define models
    model_class = getattr(
        sheet.models,
        config["model_type"],
    )
    model = model_class(
        config["model_input"],
        num_listeners=config.get("num_listeners", None),
        num_domains=config.get("num_domains", None),
        **config["model_params"],
    ).to(device)

    # define criterions
    criterion = {}
    if config["mean_score_criterions"] is not None:
        criterion["mean_score_criterions"] = [
            {
                "type": criterion_dict["criterion_type"],
                "criterion": getattr(sheet.losses, criterion_dict["criterion_type"])(
                    **criterion_dict["criterion_params"]
                ),
                "weight": criterion_dict["criterion_weight"],
            }
            for criterion_dict in config["mean_score_criterions"]
        ]
    if config.get("categorical_head_criterions", None) is not None:
        criterion["categorical_head_criterions"] = [
            {
                "type": criterion_dict["criterion_type"],
                "criterion": getattr(sheet.losses, criterion_dict["criterion_type"])(
                    **criterion_dict["criterion_params"]
                ),
                "weight": criterion_dict["criterion_weight"],
            }
            for criterion_dict in config["categorical_head_criterions"]
        ]
    if config.get("listener_score_criterions", None) is not None:
        criterion["listener_score_criterions"] = [
            {
                "type": criterion_dict["criterion_type"],
                "criterion": getattr(sheet.losses, criterion_dict["criterion_type"])(
                    **criterion_dict["criterion_params"]
                ),
                "weight": criterion_dict["criterion_weight"],
            }
            for criterion_dict in config["listener_score_criterions"]
        ]

    # define optimizers and schedulers
    optimizer_class = getattr(
        torch.optim,
        # keep compatibility
        config.get("optimizer_type", "Adam"),
    )
    optimizer = optimizer_class(
        model.parameters(),
        **config["optimizer_params"],
    )
    if config["scheduler_type"] is not None:
        scheduler = get_scheduler(
            optimizer,
            config["scheduler_type"],
            config["train_max_steps"],
            config["scheduler_params"],
        )
    else:
        scheduler = None

    if args.distributed:
        # wrap model for distributed training
        try:
            from apex.parallel import DistributedDataParallel
        except ImportError:
            raise ImportError(
                "apex is not installed. please check https://github.com/NVIDIA/apex."
            )
        model = DistributedDataParallel(model)

    # show settings
    logging.info(
        "Model parameters: {}".format(humanfriendly.format_size(model.get_num_params()))
    )
    logging.info(model)
    logging.info(optimizer)
    logging.info(scheduler)
    logging.info(criterion)

    # define trainer
    trainer_class = getattr(sheet.trainers, config["trainer_type"])
    trainer = trainer_class(
        steps=0,
        epochs=0,
        data_loader=data_loader,
        sampler=sampler,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device=device,
    )

    # load pretrained parameters from checkpoint
    if len(args.init_checkpoint) != 0:
        trainer.load_trained_modules(
            args.init_checkpoint, init_mods=config["init-mods"]
        )
        logging.info(f"Successfully load parameters from {args.init_checkpoint}.")

    # resume from checkpoint
    if len(args.resume) != 0:
        trainer.load_checkpoint(args.resume)
        logging.info(f"Successfully resumed from {args.resume}.")

    # freeze modules if necessary
    if config.get("freeze-mods", None) is not None:
        assert type(config["freeze-mods"]) is list
        trainer.freeze_modules(config["freeze-mods"])
        logging.info(f"Freeze modules with prefixes {config['freeze-mods']}.")

    # save config
    config["version"] = sheet.__version__  # add version info
    with open(os.path.join(args.outdir, "config.yml"), "w") as f:
        yaml.dump(config, f, Dumper=yaml.Dumper)
    for key, value in config.items():
        logging.info(f"{key} = {value}")

    # run training loop
    # try:
    #     trainer.run()
    # finally:
    #     trainer.save_checkpoint(
    #         os.path.join(config["outdir"], f"checkpoint-{trainer.steps}steps.pkl")
    #     )
    #     logging.info(f"Successfully saved checkpoint @ {trainer.steps}steps.")
    # NOTE(unilight): I don't think we need to save again here
    trainer.run()


if __name__ == "__main__":
    main()
