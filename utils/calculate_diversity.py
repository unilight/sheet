#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2024 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

import argparse
import numpy as np
import sheet
import sheet.datasets
import torch
from tqdm import tqdm
from s3prl.nn import S3PRLUpstream
import os

FS=16000

def get_parser():
    parser = argparse.ArgumentParser(description="calculate diversity given an input csv file.")
    parser.add_argument(
        "--s3prl_name",
        required=True,
        type=str,
        help="S3PRL upstream model name (e.g., hubert_base, wav2vec2_large_960h, etc.)",
    )
    parser.add_argument(
        "--ssl_model_layer_idx",
        type=int,
        default=-1,
        help="layer index of the SSL model to extract features (default: -1)",
    )
    parser.add_argument(
        "--csv",
        required=True,
        type=str,
        help=("csv file path to construct datastore."),
    )
    parser.add_argument(
        "--embed_path",
        required=True,
        type=str,
        help=("file path to save embeds."),
    )
    return parser


def pairwise_squared_euclidean(X):
    # X: n x d
    X_norm = np.sum(X ** 2, axis=1).reshape(-1, 1)  # (n, 1)
    D = X_norm + X_norm.T - 2 * np.dot(X, X.T)  # (n, n)
    return D

def main():
    args = get_parser().parse_args()

    # get dataset
    dataset_class = getattr(sheet.datasets, "NonIntrusiveDataset")
    dataset = dataset_class(
        csv_path=args.csv,
        target_sample_rate=FS,
        model_input="waveform",
        use_phoneme=False,
        symbols=None,
        wav_only=True,
        allow_cache=False,
    )
    print(f"Number of samples = {len(dataset)}.")

    # setup device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    if args.s3prl_name in S3PRLUpstream.available_names():
        ssl_model = S3PRLUpstream(args.s3prl_name).eval().to(device)
    else:
        raise ValueError(f"Unknown S3PRL upstream model: {args.s3prl_name}")

    # Check if embed_path exists
    if os.path.exists(args.embed_path):
        print(f"Loading embeddings from {args.embed_path}.")
        embeds = torch.load(args.embed_path)
    else:
        print(f"Embeddings not found at {args.embed_path}. Extracting embeddings.")
        # forward
        embeds = []
        with torch.no_grad(), tqdm(dataset, desc="[Embedding extraction]") as pbar:
            for batch in pbar:
                # set up model input
                model_input = batch["waveform"].unsqueeze(0).to(device)
                model_lengths = model_input.new_tensor([model_input.size(1)]).long()

                all_encoder_outputs, _ = ssl_model(model_input, model_lengths)
                embed = (
                    torch.mean(
                        all_encoder_outputs[
                            args.ssl_model_layer_idx
                        ].squeeze(0),
                        dim=0,
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )
        
                embeds.append(embed)

        embeds = np.array(embeds)
        torch.save(embeds, args.embed_path)
        print(f"Embeddings saved to {args.embed_path}.")
    
    # calculate diversity
    print("Calculating diversity...")

    # Shuffle embeddings before calculating diversity scores
    np.random.shuffle(embeds)

    # Calculate diversity scores for different percentages of embeddings
    percentages = [1, 2, 3, 4, 5, 10, 20, 25, 50, 100]
    diversity_scores = []

    for percentage in percentages:
        num_samples = int(len(embeds) * (percentage / 100))
        subset_embeds = embeds[:num_samples]
        diversity_score = np.mean(pairwise_squared_euclidean(subset_embeds))
        diversity_scores.append((percentage, diversity_score))

    # Print diversity scores
    for percentage, score in diversity_scores:
        print(f"Diversity score for {percentage}% embeddings: {score}")

if __name__ == "__main__":
    main()
