#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2024 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

""" LSTM LM model.

    This is based on:
    https://github.com/soumimaiti/speechlmscore_tool/tree/main
"""
#################################################################################################

import argparse
import logging
import os
import sys
from typing import Tuple, Union

import torch
import torch.nn as nn
import yaml

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("lstm_lm")


class LSTMLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        unit: int = 650,
        nhid: int = None,
        nlayers: int = 2,
        dropout_rate: float = 0.0,
        ignore_id: int = 0,
    ):
        super().__init__()

        ninp = unit
        if nhid is None:
            nhid = unit

        self.drop = nn.Dropout(dropout_rate)
        self.encoder = nn.Embedding(vocab_size, ninp, padding_idx=ignore_id)

        self.rnn = nn.LSTM(ninp, nhid, nlayers, dropout=dropout_rate, batch_first=True)

        self.decoder = nn.Linear(nhid, vocab_size)
        self.nhid = nhid
        self.nlayers = nlayers

    def zero_state(self):
        """Initialize LM state filled with zero values."""
        h = torch.zeros((self.nlayers, self.nhid), dtype=torch.float)
        c = torch.zeros((self.nlayers, self.nhid), dtype=torch.float)
        state = h, c

        return state

    def forward(
        self, input: torch.Tensor, hidden: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(
            output.contiguous().view(output.size(0) * output.size(1), output.size(2))
        )
        return (
            decoded.view(output.size(0), output.size(1), decoded.size(1)),
            hidden,
        )

    @classmethod
    def build_model(cls, vocab_size, config_file):
        logging.info(f"Vocabulary size: {vocab_size }")
        logging.info(f"config_file: {config_file }")

        with open(config_file, "r", encoding="utf-8") as f:
            args = yaml.safe_load(f)
            logging.info(f"args: {args }")

        args = argparse.Namespace(**args)

        logging.info(f"args: {args.lm_conf }")

        # Build LM model
        model = LSTMLM(vocab_size, **args.lm_conf)

        return model

    @classmethod
    def build_model_from_file(
        config_file,
        model_file,
        device="cpu",
    ):

        with open("r", encoding="utf-8") as f:
            args = yaml.safe_load(f)
        args = argparse.Namespace(**args)

        model = build_model(args)
        model.to(device)

        return model
