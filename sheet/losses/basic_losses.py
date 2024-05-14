#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2024 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

"""Basic losses."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from sheet.modules.utils import make_non_pad_mask


class ScalarLoss(nn.Module):
    """
    Loss for scalar output (we use the clipped MSE loss)
    """

    def __init__(self, tau, masked_loss=False):
        super(ScalarLoss, self).__init__()
        self.tau = tau
        self.masked_loss = masked_loss
        self.criterion = torch.nn.MSELoss(reduction="none")

    def forward_criterion(self, y_hat, label, criterion_module, masks=None):
        # might investigate how to combine masked loss with categorical output
        if masks is not None:
            y_hat = y_hat.masked_select(masks)
            label = label.masked_select(masks)

        y_hat = y_hat.squeeze(-1)
        loss = criterion_module(y_hat, label)
        threshold = torch.abs(y_hat - label) > self.tau
        loss = torch.mean(threshold * loss)
        return loss

    def forward(self, pred_score, gt_score, lens, device):
        """
        Args:
            pred_mean, pred_score: [batch, time, 1/5]
        """
        # make mask
        if self.masked_loss:
            masks = make_non_pad_mask(lens).to(device)
        else:
            masks = None

        # repeat for frame level loss
        time = pred_score.shape[1]
        # gt_mean = gt_mean.unsqueeze(1).repeat(1, time)
        gt_score = gt_score.unsqueeze(1).repeat(1, time)

        loss = self.forward_criterion(pred_score, gt_score, self.criterion, masks)
        return loss


#####################################################################################

# Categorical loss was not useful in initial experiments, but I keep it here for future reference


class CategoricalLoss(nn.Module):
    def __init__(self, alpha, lamb):
        super(CategoricalLoss, self).__init__()
        self.alpha = alpha
        self.lamb = lamb

        if self.alpha > 0:
            self.mean_net_criterion = nn.CrossEntropyLoss()
        self.main_criterion = nn.CrossEntropyLoss()

    def ce(self, y_hat, label, criterion, masks=None):
        if masks is not None:
            y_hat = y_hat.masked_select(masks)
            label = label.masked_select(masks)
        ce = criterion(y_hat, label - 1)
        return ce

    def forward(self, pred_mean, gt_mean, pred_score, gt_score, lens, device):
        # make mask
        if self.masked_loss:
            masks = make_non_pad_mask(lens).to(device)
        else:
            masks = None

        score_ce = self.ce(pred_score, gt_score, self.main_criterion, masks)
        if self.alpha > 0:
            mean_ce = self.ce(pred_mean, gt_mean, self.mean_net_criterion, masks)
            return self.alpha * mean_ce + self.lamb * score_ce, mean_ce, score_ce
        else:
            return self.lamb * score_ce, None, score_ce
