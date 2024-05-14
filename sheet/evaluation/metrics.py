# -*- coding: utf-8 -*-

# Copyright 2024 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

"""Script to calculate metrics."""

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
