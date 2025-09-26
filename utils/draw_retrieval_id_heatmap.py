#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2025 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

"""
The `domain_id_knn_1` inference method in AlignNet retrieves the domain ID
of the training sample that is closest to the input sample in the SSL encoder space.

This script draws the retrieved ID.
"""

import argparse
import csv
import sys
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

times_new_roman_bold = fm.FontProperties(
    fname=r"/home/z44476r/Times New Roman Bold.ttf"
)
fontsize = 35
large_font = 30

train_sets = [
    "BVCC",
    "SOMOS",
    "SingMOS",
    "NISQA",
    "TMHINT-QI",
    "Tencent",
    "PSTN",
    "URGENT2024\n-MOS"
]
synthetic_sets = [
    "bvcc_test",
    "somos_test",
    "bc19_test",
    "bc23a_test",
    "bc23b_test",
    "svcc23_test",
    "singmos_test",
    "brspeechmos_test",
    "hablamos_test",
    "ttsds2_test",
]

distorted_sets = [
    "nisqa_FOR",
    "nisqa_LIVETALK",
    "nisqa_P501",
    "tmhintqi_test",
    "tmhint_qi_s_test",
    "tcd_voip_test",
    "vmc24_track3_test",
]

order = [
    "bvcc_test",
    "somos_test",
    "bc19_test",
    "bc23a_test",
    "bc23b_test",
    "svcc23_test",
    "singmos_test",
    "brspeechmos_test",
    "hablamos_test",
    "ttsds2_test",
    "nisqa_FOR",
    "nisqa_LIVETALK",
    "nisqa_P501",
    "tmhintqi_test",
    "tmhint_qi_s_test",
    "tcd_voip_test",
    "vmc24_track3_test",
]

test_set_names = [
    "BVCC",
    "SOMOS",
    "BC19",
    "BC23a",
    "BC23b",
    "SVCC23",
    "SingMOS",
    "BRSpeechMOS",
    "HablaMOS",
    "TTSDS2",
    "NISQA FOR",
    "NISQA LIVETALK",
    "NISQA P501",
    "TMHINT-QI",
    "TMHINT-QI(S)",
    "TCD-VOIP",
    "VMC'24 Track 3",
]

NUM_TRAIN_SETS = len(train_sets)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dir",
        type=str,
    )
    args = parser.parse_args()

    id_values = list(range(NUM_TRAIN_SETS))
    data = []

    for d in order:
        results_filepath = os.path.join(args.dir, d, "results.csv")
        if os.path.isfile(results_filepath):
            with open(results_filepath, newline="") as csvfile:
                reader = csv.DictReader(csvfile)
                contents = [line for line in reader]
            counts = [0 for _ in range(NUM_TRAIN_SETS)]
            for item in contents:
                _id = int(item["retrieved_id"])
                counts[_id] += 1
            row = [counts[i] / len(contents) for i in range(NUM_TRAIN_SETS)]
            data.append(row)

    # Create DataFrame
    df = pd.DataFrame(data, columns=train_sets)
    df.index = [d for d in test_set_names]

    # Plot using seaborn heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(
        df,
        annot=True,
        fmt=".1%",
        cbar=False,
        cmap=sns.cubehelix_palette(as_cmap=True),
        linewidth=0.5,
        annot_kws={"fontsize": 14},
    )
    plt.title("Ratio of retrieved dataset embedding in each test set", fontsize=18)
    plt.ylabel("Test Set", fontsize=16)
    plt.xlabel("Retrieved dataset embedding", fontsize=16)

    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)

    plt.tight_layout()
    plt.savefig(os.path.join(args.dir, "retrieval_id.png"), dpi=100)


if __name__ == "__main__":
    main()
