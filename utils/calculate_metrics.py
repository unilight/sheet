#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2024 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

import argparse
from collections import defaultdict
import csv
import numpy as np
import scipy

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
    parser = argparse.ArgumentParser(description="calculate metrics given an input csv file.")
    parser.add_argument("--csv", required=True, type=str, help="input csv file")
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
        answer = float(item["answer"])
        avg_score = float(item["avg_score"])
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


if __name__ == "__main__":
    main()