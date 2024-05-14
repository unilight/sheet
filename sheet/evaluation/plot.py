# -*- coding: utf-8 -*-

# Copyright 2024 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

"""Script to plot figures."""

import matplotlib
import numpy as np

# Force matplotlib to not use any Xwindows backend.
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STYLE = "seaborn-v0_8-deep"


def plot_utt_level_hist(true_mean_scores, predict_mean_scores, filename):
    """Plot utterance-level histrogram.

    Args:
        true_mean_scores: ndarray of true scores
        predict_mean_scores: ndarray of predicted scores
        filename: name of the saved figure
    """
    plt.style.use(STYLE)
    bins = np.linspace(1, 5, 40)
    plt.figure(2)
    plt.hist(
        [true_mean_scores, predict_mean_scores], bins, label=["true_mos", "predict_mos"]
    )
    plt.legend(loc="upper right")
    plt.xlabel("MOS")
    plt.ylabel("number")
    plt.show()
    plt.savefig(filename, dpi=150)
    plt.close()


def plot_utt_level_scatter(
    true_mean_scores, predict_mean_scores, filename, LCC, SRCC, MSE, KTAU
):
    """Plot utterance-level scatter plot

    Args:
        true_mean_scores: ndarray of true scores
        predict_mean_scores: ndarray of predicted scores
        filename: name of the saved figure
        LCC, SRCC, MSE, KTAU: metrics to be shown on the figure
    """
    M = np.max([np.max(predict_mean_scores), 5])
    plt.figure(3)
    plt.scatter(
        true_mean_scores,
        predict_mean_scores,
        s=15,
        color="b",
        marker="o",
        edgecolors="b",
        alpha=0.20,
    )
    plt.xlim([0.5, M])
    plt.ylim([0.5, M])
    plt.xlabel("True MOS")
    plt.ylabel("Predicted MOS")
    plt.title(
        "Utt level LCC= {:.4f}, SRCC= {:.4f}, MSE= {:.4f}, KTAU= {:.4f}".format(
            LCC, SRCC, MSE, KTAU
        )
    )
    plt.show()
    plt.savefig(filename, dpi=150)
    plt.close()


def plot_sys_level_scatter(
    true_sys_mean_scores, predict_sys_mean_scores, filename, LCC, SRCC, MSE, KTAU
):
    """Plot system-level scatter plot

    Args:
        true_sys_mean_scores: ndarray of true system level scores
        predict_sys_mean_scores: ndarray of predicted system level scores
        filename: name of the saved figure
        LCC, SRCC, MSE, KTAU: metrics to be shown on the figure
    """
    M = np.max([np.max(predict_sys_mean_scores), 5])
    plt.figure(3)
    plt.scatter(
        true_sys_mean_scores,
        predict_sys_mean_scores,
        s=15,
        color="b",
        marker="o",
        edgecolors="b",
    )
    plt.xlim([0.5, M])
    plt.ylim([0.5, M])
    plt.xlabel("True MOS")
    plt.ylabel("Predicted MOS")
    plt.title(
        "Sys level LCC= {:.4f}, SRCC= {:.4f}, MSE= {:.4f}, KTAU= {:.4f}".format(
            LCC, SRCC, MSE, KTAU
        )
    )
    plt.show()
    plt.savefig(filename, dpi=150)
    plt.close()
