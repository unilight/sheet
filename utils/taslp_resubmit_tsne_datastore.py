#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2025 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

"""
2025.09 for TASLP resubmission
Take a datastore as input and do t-SNE
"""

import argparse
import csv
import os
from collections import defaultdict
from tqdm import tqdm

import matplotlib
import numpy as np
import scipy
import h5py

import random
from sklearn.manifold import TSNE

# Force matplotlib to not use any Xwindows backend.
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STYLE = "seaborn-v0_8-deep"

train_sets = [
    "BVCC",
    "SOMOS",
    "SingMOS",
    "NISQA",
    "TMHINT-QI",
    "Tencent",
    "PSTN",
    "URGENT2024-MOS"
]

def get_parser():
    parser = argparse.ArgumentParser(
        description="Do t-SNE visualization given a datstore."
    )
    parser.add_argument("--datastore", required=True, type=str, help="input csv file")
    return parser


def main():
    args = get_parser().parse_args()

    embeds = []
    scores = []
    paths = []

    with h5py.File(args.datastore, "r") as f:
        for hdf5_path in tqdm(list(f["scores"].keys())):
            paths.append(hdf5_path)
            embeds.append(f["embeds"][hdf5_path][()])
            scores.append(f["scores"][hdf5_path][()])

    # Step 1: group embeddings by dataset id
    dataset_to_embeds = {i: [] for i in range(8)}
    for embed, path in zip(embeds, paths):
        dataset_id = int(path[0])
        dataset_to_embeds[dataset_id].append(embed)

    # Step 2: sample 1000 from each dataset
    sampled_embeds = []
    sampled_labels = []
    for dataset_id, embs in dataset_to_embeds.items():
        chosen = random.sample(embs, 1000)  # uniform random
        sampled_embeds.extend(chosen)
        sampled_labels.extend([dataset_id] * 1000)

    sampled_embeds = np.stack(sampled_embeds)  # shape: (8000, dim)
    sampled_labels = np.array(sampled_labels)

    # Step 3: t-SNE
    print("Running t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, init="pca", learning_rate="auto")
    embeds_2d = tsne.fit_transform(sampled_embeds)
    print("Done.")

    # Step 4: visualize
    plot_path = os.path.join(os.path.dirname(args.datastore), f"tsne")
    plt.figure(figsize=(10, 8))
    
    # scatter = plt.scatter(
    #     embeds_2d[:, 0], embeds_2d[:, 1],
    #     c=sampled_labels, cmap="tab10", alpha=0.7, s=10
    # )
    # plt.colorbar(scatter, ticks=range(8), label="Dataset ID")
    colors = plt.cm.tab10.colors  # 10 distinct colors
    for dataset_id in range(8):
        mask = sampled_labels == dataset_id
        plt.scatter(
            embeds_2d[mask, 0],
            embeds_2d[mask, 1],
            s=20,
            alpha=0.7,
            color=colors[dataset_id],
            label=train_sets[dataset_id]
        )

    plt.legend(
        # title="Datasets",
        fontsize=14,
    )
    plt.xticks([])
    plt.yticks([])
    
    # plt.title("t-SNE of 8 datasets (1000 samples each)")
    plt.tight_layout()
    plt.savefig(plot_path + ".png", dpi=150)
    plt.savefig(plot_path + ".pdf", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
