#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2024 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

"""NLL losses."""

import torch
import torch.nn as nn
from sheet.modules.utils import make_non_pad_mask


class GaussianNLLLoss(nn.Module):
    """
    Gaussian NLL loss (for uncertainty modeling)
    """

    def __init__(self, tau, masked_loss=False):
        super(GaussianNLLLoss, self).__init__()
        self.tau = tau
        self.masked_loss = masked_loss

    def forward_criterion(self, y_hat, logvar, label, masks=None):
        """
        loss = 0.5 * (precision * (target - mean) ** 2 + log_var)
        """

        # might investigate how to combine masked loss with categorical output
        if masks is not None:
            y_hat = y_hat.masked_select(masks)
            logvar = logvar.masked_select(masks)
            label = label.masked_select(masks)

        y_hat = y_hat.squeeze(-1)
        logvar = logvar.squeeze(-1)
        precision = torch.exp(-logvar)
        loss = 0.5 * (precision * (y_hat - label) ** 2 + logvar)
        threshold = torch.abs(y_hat - label) > self.tau
        loss = torch.mean(threshold * loss)
        return loss

    def forward(self, pred_score, pred_logvar, gt_score, device, lens=None):
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

        loss = self.forward_criterion(pred_score, pred_logvar, gt_score, masks)
        return loss


class LaplaceNLLLoss(nn.Module):
    """
    Laplace NLL loss (for uncertainty modeling)
    """

    def __init__(self, tau, masked_loss=False):
        super(LaplaceNLLLoss, self).__init__()
        self.tau = tau
        self.masked_loss = masked_loss

    def forward_criterion(self, y_hat, logvar, label, masks=None):
        """
        loss = 0.5 * (precision * (target - mean) ** 2 + log_var)
        """

        # might investigate how to combine masked loss with categorical output
        if masks is not None:
            y_hat = y_hat.masked_select(masks)
            logvar = logvar.masked_select(masks)
            label = label.masked_select(masks)

        y_hat = y_hat.squeeze(-1)
        logvar = logvar.squeeze(-1)
        b = torch.exp(logvar) + 1e-6
        loss = torch.abs(y_hat - label) / b + logvar
        threshold = torch.abs(y_hat - label) > self.tau
        loss = torch.mean(threshold * loss)
        return loss

    def forward(self, pred_score, pred_logvar, gt_score, device, lens=None):
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

        loss = self.forward_criterion(pred_score, pred_logvar, gt_score, masks)
        return loss
