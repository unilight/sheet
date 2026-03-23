#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2024 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

""" Unit LM model.

    This is based on:
    https://github.com/soumimaiti/speechlmscore_tool/tree/main
"""
#################################################################################################

import logging
import os
import sys
from pathlib import Path

# The langugae model code is inspired by code from espnet language model code
#  https://github.com/espnet/espnet
#
from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from sheet.modules.speechlmscore.lstm_lm import LSTMLM


def make_pad_mask(lengths, xs=None, length_dim=-1, maxlen=None):

    if length_dim == 0:
        raise ValueError("length_dim cannot be 0: {}".format(length_dim))

    if not isinstance(lengths, list):
        lengths = lengths.long().tolist()

    bs = int(len(lengths))
    if maxlen is None:
        if xs is None:
            maxlen = int(max(lengths))
        else:
            maxlen = xs.size(length_dim)
    else:
        assert xs is None
        assert maxlen >= int(max(lengths))

    seq_range = torch.arange(0, maxlen, dtype=torch.int64)
    seq_range_expand = seq_range.unsqueeze(0).expand(bs, maxlen)
    seq_length_expand = seq_range_expand.new(lengths).unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand

    if xs is not None:
        assert xs.size(0) == bs, (xs.size(0), bs)

        if length_dim < 0:
            length_dim = xs.dim() + length_dim

        ind = tuple(
            slice(None) if i in (0, length_dim) else None for i in range(xs.dim())
        )
        mask = mask[ind].expand_as(xs).to(xs.device)
    return mask


class LM(nn.Module):
    def __init__(self, lm, vocab_size, ignore_id=0):
        super().__init__()
        self.lm = lm
        self.sos = vocab_size - 1
        self.eos = vocab_size - 1
        self.ignore_id = ignore_id

    def nll(
        self,
        text,
        text_lengths,
        max_length=None,
    ):
        """Compute negative log likelihood(nll)
        Normally, this function is called in batchify_nll.
        Args:
            text: (Batch, Length)
            text_lengths: (Batch,)
            max_lengths: int
        """
        batch_size = text.size(0)

        # For data parallel
        if max_length is None:
            text = text[:, : text_lengths.max()]
        else:
            text = text[:, :max_length]

        # 1. Create a sentence pair like '<sos> w1 w2 w3' and 'w1 w2 w3 <eos>'
        # text: (Batch, Length) -> x, y: (Batch, Length + 1)
        x = F.pad(text, [1, 0], "constant", self.eos)
        t = F.pad(text, [0, 1], "constant", self.ignore_id)
        for i, l in enumerate(text_lengths):
            t[i, l] = self.sos
        x_lengths = text_lengths + 1

        # 2. Forward Language model
        # x: (Batch, Length) -> y: (Batch, Length, NVocab)
        y, _ = self.lm(x, None)

        # 3. Calc negative log likelihood
        # nll: (BxL,)
        nll = F.cross_entropy(y.view(-1, y.shape[-1]), t.view(-1), reduction="none")
        # nll: (BxL,) -> (BxL,)
        if max_length is None:
            nll.masked_fill_(make_pad_mask(x_lengths).to(nll.device).view(-1), 0.0)
        else:
            nll.masked_fill_(
                make_pad_mask(x_lengths, maxlen=max_length + 1).to(nll.device).view(-1),
                0.0,
            )
        # nll: (BxL,) -> (B, L)
        nll = nll.view(batch_size, -1)

        return nll, x_lengths, y

    @classmethod
    def build_model(cls, token_list_inp, config_file, lm_type="lstm"):

        if isinstance(token_list_inp, str):
            with open(token_list_inp, encoding="utf-8") as f:
                token_list = [line.rstrip() for line in f]

            token_list = token_list.copy()
        elif isinstance(token_list_inp, (tuple, list)):
            token_list = token_list_inp.copy()
        else:
            raise RuntimeError("token_list must be str or dict")

        logging.info(f"token_list: {token_list }")
        vocab_size = len(token_list)
        logging.info(f"Vocabulary size: {vocab_size }")

        # 1. Build LM model
        if lm_type == "lstm":
            lm = LSTMLM.build_model(vocab_size, config_file)

        # Assume the last-id is sos_and_eos
        model = LM(lm=lm, vocab_size=vocab_size)

        return model

    # ~~~~~~~~~ The methods below are mainly used for inference ~~~~~~~~~
    @classmethod
    def build_model_from_file(
        cls, token_list_inp, config_file, model_file, device="cpu", lm_type="lstm"
    ):
        model = cls.build_model(token_list_inp, config_file, lm_type)
        model.to(device)

        # load state dict into lm
        model.load_state_dict(torch.load(model_file, map_location=device))

        return model
