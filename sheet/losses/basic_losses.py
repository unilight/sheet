#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2024 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

"""Basic losses."""

import torch
import torch.nn as nn
from sheet.modules.utils import make_non_pad_mask


class ScalarLoss(nn.Module):
    """
    Loss for scalar output (we use the clipped MSE loss)
    """

    def __init__(self, tau, order=2, masked_loss=False):
        super(ScalarLoss, self).__init__()
        self.tau = tau
        self.masked_loss = masked_loss
        if order == 2:
            self.criterion = torch.nn.MSELoss(reduction="none")
        elif order == 1:
            self.criterion = torch.nn.L1Loss(reduction="none")
        else:
            raise NotImplementedError

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

    def forward(self, pred_score, gt_score, device, lens=None):
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


class CategoricalLoss(nn.Module):
    def __init__(self, masked_loss=False):
        super(CategoricalLoss, self).__init__()
        self.masked_loss = masked_loss
        self.criterion = nn.CrossEntropyLoss(reduction="none")

    def ce(self, y_hat, label, criterion, masks=None):
        if masks is not None:
            y_hat = y_hat.masked_select(masks)
            label = label.masked_select(masks)

        # y_hat must have shape (batch_size, num_classes, ...)
        y_hat = y_hat.permute(0, 2, 1)

        ce = criterion(y_hat, label)
        return torch.mean(ce)

    def forward(self, pred_score, gt_score, lens, device):
        # make mask
        if self.masked_loss:
            masks = make_non_pad_mask(lens).to(device)
        else:
            masks = None

        # repeat for frame level loss
        time = pred_score.shape[1]
        gt_score = gt_score.unsqueeze(1).repeat(1, time).type(torch.long)

        score_ce = self.ce(pred_score, gt_score, self.criterion, masks)
        return score_ce
