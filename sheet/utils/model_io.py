#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2024 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

import logging
import os
from collections import OrderedDict

import torch


def print_new_keys(state_dict, modules, model_path):
    logging.info(f"Loading {modules} from model: {model_path}")

    for k in state_dict.keys():
        logging.warning(f"Overriding module {k}")


def filter_modules(model_state_dict, modules):
    """Filter non-matched modules in model state dict.
    Args:
        model_state_dict (Dict): Pre-trained model state dict.
        modules (List): Specified module(s) to transfer.
    Return:
        new_mods (List): Filtered module list.
    """
    new_mods = []
    incorrect_mods = []

    mods_model = list(model_state_dict.keys())
    for mod in modules:
        if any(key.startswith(mod) for key in mods_model):
            new_mods += [mod]
        else:
            incorrect_mods += [mod]

    if incorrect_mods:
        logging.error(
            "Specified module(s) don't match or (partially match) "
            f"available modules in model. You specified: {incorrect_mods}."
        )
        logging.error("The existing modules in model are:")
        logging.error(f"{mods_model}")
        exit(1)

    return new_mods


def get_partial_state_dict(model_state_dict, modules):
    """Create state dict with specified modules matching input model modules.
    Args:
        model_state_dict (Dict): Pre-trained model state dict.
        modules (Dict): Specified module(s) to transfer.
    Return:
        new_state_dict (Dict): State dict with specified modules weights.
    """
    new_state_dict = OrderedDict()

    for key, value in model_state_dict.items():
        if any(key.startswith(m) for m in modules):
            new_state_dict[key] = value

    return new_state_dict


def transfer_verification(model_state_dict, partial_state_dict, modules):
    """Verify tuples (key, shape) for input model modules match specified modules.
    Args:
        model_state_dict (Dict) : Main model state dict.
        partial_state_dict (Dict): Pre-trained model state dict.
        modules (List): Specified module(s) to transfer.
    Return:
        (bool): Whether transfer learning is allowed.
    """
    model_modules = []
    partial_modules = []

    for key_m, value_m in model_state_dict.items():
        if any(key_m.startswith(m) for m in modules):
            model_modules += [(key_m, value_m.shape)]
    model_modules = sorted(model_modules, key=lambda x: (x[0], x[1]))

    for key_p, value_p in partial_state_dict.items():
        if any(key_p.startswith(m) for m in modules):
            partial_modules += [(key_p, value_p.shape)]
    partial_modules = sorted(partial_modules, key=lambda x: (x[0], x[1]))

    module_match = model_modules == partial_modules

    if not module_match:
        logging.error(
            "Some specified modules from the pre-trained model "
            "don't match with the new model modules:"
        )
        logging.error(f"Pre-trained: {set(partial_modules) - set(model_modules)}")
        logging.error(f"New model: {set(model_modules) - set(partial_modules)}")
        exit(1)

    return module_match


def freeze_modules(model, modules):
    """Freeze model parameters according to modules list.
    Args:
        model (torch.nn.Module): Main model.
        modules (List): Specified module(s) to freeze.
    Return:
        model (torch.nn.Module) : Updated main model.
        model_params (filter): Filtered model parameters.
    """
    for mod, param in model.named_parameters():
        if any(mod.startswith(m) for m in modules):
            logging.warning(f"Freezing {mod}. It will not be updated during training.")
            param.requires_grad = False

    model_params = filter(lambda x: x.requires_grad, model.parameters())

    return model, model_params


@torch.no_grad()
def model_average(model, outdir):
    """Generate averaged model from existing models

    Args:
        model: the model instance
        outdir: the directory contains the model files
    """
    # get model checkpoints
    checkpoint_paths = [
        os.path.join(outdir, p)
        for p in os.listdir(outdir)
        if os.path.isfile(os.path.join(outdir, p)) and p.endswith("steps.pkl")
    ]
    n = len(checkpoint_paths)

    # load the checkpoints
    avg = None
    for checkpoint_path in checkpoint_paths:
        states = torch.load(checkpoint_path, map_location="cpu")["model"]
        if avg is None:
            avg = states
        else:
            # Accumulated
            for k in avg:
                avg[k] = avg[k] + states[k]

    # take average
    for k in avg:
        if str(avg[k].dtype).startswith("torch.int"):
            # For int type, not averaged, but only accumulated.
            # e.g. BatchNorm.num_batches_tracked
            # (If there are any cases that requires averaging
            #  or the other reducing method, e.g. max/min, for integer type,
            #  please report.)
            logging.info(f"Accumulating {k} instead of averaging")
            pass
        else:
            avg[k] = avg[k] / n

    # load into model
    model.load_state_dict(avg)

    return model, checkpoint_paths
