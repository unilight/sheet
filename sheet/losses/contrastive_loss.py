#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2024 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

"""Contrastive loss proposed in UTMOS."""

import torch
import torch.nn as nn


class ContrastiveLoss(nn.Module):
    '''
    Contrastive Loss
    Args:
        margin: non-neg value, the smaller the stricter the loss will be, default: 0.2        
        
    '''
    def __init__(self, margin=0.2):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    
    def forward(self, pred_score, gt_score, lens, device):
        if pred_score.dim() > 2:
            pred_score = pred_score.mean(dim=1).squeeze(1)
        # pred_score, gt_score: tensor, [batch_size]  

        gt_diff = gt_score.unsqueeze(1) - gt_score.unsqueeze(0)
        pred_diff = pred_score.unsqueeze(1) - pred_score.unsqueeze(0)
        
        loss = torch.maximum(torch.zeros(gt_diff.shape).to(gt_diff.device), torch.abs(pred_diff - gt_diff) - self.margin) 
        loss = loss.mean().div(2)

        return loss