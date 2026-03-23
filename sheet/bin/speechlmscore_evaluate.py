#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2024 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

"""Dump feature for SpeechLMScore calculation ."""


import argparse
import csv
import logging
import os

import h5py
import joblib
import numpy as np
import torch

# from transformers import HubertModel
from s3prl.nn import S3PRLUpstream
from sheet.datasets import NonIntrusiveDataset
from sheet.modules.speechlmscore.lm import LM
from tqdm import tqdm

SAMPLING_RATE = 16000


class KM(object):
    def __init__(self, km_path, device="cpu"):
        self.km_model = joblib.load(km_path)
        self.C_np = self.km_model.cluster_centers_.transpose()
        self.Cnorm_np = (self.C_np**2).sum(0, keepdims=True)

        self.C = torch.from_numpy(self.C_np).to(device)
        self.Cnorm = torch.from_numpy(self.Cnorm_np).to(device)

    def __call__(self, x):
        if isinstance(x, torch.Tensor):
            dist = (
                x.pow(2).sum(1, keepdim=True) - 2 * torch.matmul(x, self.C) + self.Cnorm
            )
            return dist.argmin(dim=1).cpu().numpy()
        else:
            dist = (
                (x**2).sum(1, keepdims=True)
                - 2 * np.matmul(x, self.C_np)
                + self.Cnorm_np
            )
            return np.argmin(dist, axis=1)


class TokenIDConverter:
    def __init__(
        self,
        tokens_filepath,
        unk_symbol="<unk>",
    ):
        self.token_list = []
        with open(tokens_filepath, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.rstrip()
                self.token_list.append(line)

        self.token2id = {}
        for i, t in enumerate(self.token_list):
            if t in self.token2id:
                raise RuntimeError(f'Symbol "{t}" is duplicated')
            self.token2id[t] = i

        self.unk_symbol = unk_symbol
        if self.unk_symbol not in self.token2id:
            raise RuntimeError(
                f"Unknown symbol '{unk_symbol}' doesn't exist in the token_list"
            )
        self.unk_id = self.token2id[self.unk_symbol]

    def get_num_vocabulary_size(self):
        return len(self.token_list)

    def tokens2ids(self, tokens):
        return [self.token2id.get(str(i), self.unk_id) for i in tokens]


def main():
    """Construct datastore."""
    parser = argparse.ArgumentParser(
        description=(
            "Construct datastore with ssl_model in trained model "
            "(See detail in bin/construct_datastore.py)."
        )
    )
    parser.add_argument(
        "--csv-path",
        required=True,
        type=str,
        help=("csv file path to construct datastore."),
    )
    # parser.add_argument(
    #     "--hubert-model-path",
    #     type=str,
    #     help="HuBERT model checkpoint file.",
    # )
    parser.add_argument(
        "--km-model-path",
        type=str,
        help="K-means model file.",
    )
    parser.add_argument(
        "--ulm-token-list",
        type=str,
        help="Unit LM token list.",
    )
    parser.add_argument(
        "--ulm-config",
        type=str,
        help="Unit LM config.",
    )
    parser.add_argument(
        "--ulm-model-path",
        type=str,
        help="Unit LM model file.",
    )
    parser.add_argument(
        "--layer",
        type=int,
        help="Which layer of the HuBERT model to extract.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        required=True,
        help="output dir to save results.",
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

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # get dataset
    dataset = NonIntrusiveDataset(
        csv_path=args.csv_path,
        target_sample_rate=SAMPLING_RATE,
        wav_only=True,
        allow_cache=False,
    )
    logging.info(f"Number of inference samples = {len(dataset)}.")

    # load HuBERT model
    hubert_model = S3PRLUpstream("hubert_base").eval().to(device)

    # load K-means model
    km_model = KM(args.km_model_path, device=device)

    # Load unit-lm
    ulm_model = LM.build_model_from_file(
        args.ulm_token_list, args.ulm_config, args.ulm_model_path, device
    )
    token_id_converter = TokenIDConverter(args.ulm_token_list)
    print(token_id_converter.token2id)

    logging.info("Running inference...")
    with torch.no_grad():
        for batch in tqdm(dataset):
            system_id = batch["system_id"]
            sample_id = batch["sample_id"]

            model_input = batch["waveform"].unsqueeze(0).to(device)
            model_lengths = model_input.new_tensor([model_input.size(1)]).long()

            # extract hubert features
            all_encoder_outputs, _ = hubert_model(model_input, model_lengths)
            feat = all_encoder_outputs[args.layer].squeeze(0)  # .cpu().numpy()

            # run km model
            tokens = km_model(feat).tolist()

            # token to id
            token_ids = token_id_converter.tokens2ids(tokens)
            token_id_lengths = torch.tensor([len(token_ids)], dtype=torch.int)
            token_ids = (
                torch.from_numpy(np.array(token_ids, dtype=np.int64))
                .unsqueeze(0)
                .to(device)
            )

            # run ulm
            nll, lengths, hypo_lprobs = ulm_model.nll(token_ids, token_id_lengths)

            nll = nll.detach().cpu().numpy().sum(1)
            # lprobs = hypo_lprobs.detach().cpu().numpy().sum(1)
            lengths = lengths.detach().cpu().numpy()

            answer = (nll / lengths)[0]
            # answer = (lprobs / lengths)[0]
            dataset.fill_answer(batch["sample_id"], answer)

    # write results
    results = dataset.return_results()
    results_path = os.path.join(args.outdir, "results.csv")
    fieldnames = list(results[0].keys())
    with open(results_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for line in results:
            writer.writerow(line)


if __name__ == "__main__":
    main()
