#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2025 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

"""
2025.09 for TASLP resubmission
Redraw the `utt_scatter_plot.png`
"""

import argparse
import csv
import os
from collections import defaultdict

import matplotlib
import numpy as np
import scipy

# Force matplotlib to not use any Xwindows backend.
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STYLE = "seaborn-v0_8-deep"

TRAIN_SET_DICT={
    "bvcc": "BVCC",
    "tmhint-qi": "TMHINT-QI",
    "tencent": "Tencent",
    "singmos": "SingMOS",
    "somos": "SOMOS",
    "nisqa": "NISQA",
    "pstn": "PSTN",
}

TEST_SET_DICT={
    "somos_test": "SOMOS",
    "singmos_test": "SingMOS",
    "hablamos_test": "HablaMOS",
    "nisqa_P501": "NISQA P501",
    "bvcc_test": "BVCC",
    "ttsds2_test": "TTSDS2",
    "nisqa_FOR": "NISQA FOR",
    "tcd_voip_test": "TCD-VOIP",
    "tmhintqi_test": "TMHINT-QI",
    "hablamos_test": "HablaMOS",
    "bc23a_test": "BC23a",
    "bc23b_test": "BC23b",
}

def plot_utt_level_scatter(true_mean_scores, predict_mean_scores, filename, MSE, SRCC, training_set, testing_set):
    """Plot utterance-level scatter plot

    Args:
        true_mean_scores: ndarray of true scores
        predict_mean_scores: ndarray of predicted scores
        filename: name of the saved figure
        LCC, SRCC, MSE, KTAU: metrics to be shown on the figure
    """
    plt.figure(figsize=(6,6), constrained_layout=True)
    plt.xlim([1.0, 5.0])
    plt.ylim([1.0, 5.0])
    plt.xticks([1, 5])
    plt.yticks([1, 5])
    plt.plot([1.0, 5.0], [1.0, 5.0], ls="--", c=".3")
    plt.xlabel("True MOS", fontsize=20)
    plt.ylabel("Predicted MOS", fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.title("Training set: {}, testing set: {}\nUtterance-level MSE = {:.3f}, SRCC = {:.3f}".format(training_set, testing_set, MSE, SRCC), fontsize=22, pad=10)
    
    plt.scatter(
        true_mean_scores,
        predict_mean_scores,
        s=80,
        color="b",
        marker="o",
        edgecolors="b",
        alpha=0.50,
    )
    
    # plt.show()
    # plt.tight_layout()
    plt.savefig(filename.replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
    plt.savefig(filename.replace(".png", ".pdf"), bbox_inches="tight", dpi=150)
    plt.close()


def calculate(
    true_mean_scores, predict_mean_scores, true_sys_mean_scores, predict_sys_mean_scores
):

    utt_MSE = np.mean((true_mean_scores - predict_mean_scores) ** 2)
    utt_LCC = np.corrcoef(true_mean_scores, predict_mean_scores)[0][1]
    utt_SRCC = scipy.stats.spearmanr(true_mean_scores, predict_mean_scores)[0]
    utt_KTAU = scipy.stats.kendalltau(true_mean_scores, predict_mean_scores)[0]
    sys_MSE = np.mean((true_sys_mean_scores - predict_sys_mean_scores) ** 2)
    sys_LCC = np.corrcoef(true_sys_mean_scores, predict_sys_mean_scores)[0][1]
    sys_SRCC = scipy.stats.spearmanr(true_sys_mean_scores, predict_sys_mean_scores)[0]
    sys_KTAU = scipy.stats.kendalltau(true_sys_mean_scores, predict_sys_mean_scores)[0]

    return {
        "utt_MSE": utt_MSE,
        "utt_LCC": utt_LCC,
        "utt_SRCC": utt_SRCC,
        "utt_KTAU": utt_KTAU,
        "sys_MSE": sys_MSE,
        "sys_LCC": sys_LCC,
        "sys_SRCC": sys_SRCC,
        "sys_KTAU": sys_KTAU,
    }


def get_parser():
    parser = argparse.ArgumentParser(
        description="calculate metrics given an input csv file."
    )
    parser.add_argument("--csv", required=True, type=str, help="input csv file")
    parser.add_argument(
        "--answer_column",
        default="answer",
        type=str,
        help="the column that stores predicted scores. default to be `answer`",
    )
    parser.add_argument(
        "--gt_column",
        default="avg_score",
        type=str,
        help="the column that stores GT scores. default to be `avg_score`",
    )
    return parser


def main():
    args = get_parser().parse_args()

    with open(args.csv, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        contents = [line for line in reader]

    eval_results = defaultdict(list)
    eval_sys_results = defaultdict(lambda: defaultdict(list))

    for item in contents:
        sys_name = item["system_id"]
        answer = float(item[args.answer_column])
        avg_score = float(item[args.gt_column])
        eval_results["pred_mean_scores"].append(answer)
        eval_results["true_mean_scores"].append(avg_score)

        eval_sys_results["pred_mean_scores"][sys_name].append(answer)
        eval_sys_results["true_mean_scores"][sys_name].append(avg_score)

    eval_results["true_mean_scores"] = np.array(eval_results["true_mean_scores"])
    eval_results["pred_mean_scores"] = np.array(eval_results["pred_mean_scores"])
    eval_sys_results["true_mean_scores"] = np.array(
        [np.mean(scores) for scores in eval_sys_results["true_mean_scores"].values()]
    )
    eval_sys_results["pred_mean_scores"] = np.array(
        [np.mean(scores) for scores in eval_sys_results["pred_mean_scores"].values()]
    )

    # calculate metrics
    results = calculate(
        eval_results["true_mean_scores"],
        eval_results["pred_mean_scores"],
        eval_sys_results["true_mean_scores"],
        eval_sys_results["pred_mean_scores"],
    )
    print(
        f'[UTT][ MSE = {results["utt_MSE"]:.3f} | LCC = {results["utt_LCC"]:.3f} | SRCC = {results["utt_SRCC"]:.3f} | KTAU = {results["utt_KTAU"]:.3f} ] [SYS][ MSE = {results["sys_MSE"]:.3f} | LCC = {results["sys_LCC"]:.4f} | SRCC = {results["sys_SRCC"]:.4f}  | KTAU = {results["sys_KTAU"]:.3f} ]'
    )

    # draw
    parts = args.csv.split("/")
    training_set = TRAIN_SET_DICT[parts[-7]]
    testing_set = TEST_SET_DICT[parts[-2]]
    new_plot_path = os.path.join(os.path.dirname(args.csv), f"{training_set}-{testing_set}-utt_scatter_redraw.pdf")
    plot_utt_level_scatter(
        eval_results["true_mean_scores"],
        eval_results["pred_mean_scores"],
        new_plot_path,
        results["utt_MSE"],
        results["utt_SRCC"],
        training_set, testing_set
    )


if __name__ == "__main__":
    main()
