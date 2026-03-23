#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2025 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

""" AttentiveStatisticsPooling class. Based on:
    https://speechbrain.readthedocs.io/en/latest/_modules/speechbrain/lobes/models/ECAPA_TDNN.html#AttentiveStatisticsPooling

"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def length_to_mask(lengths, max_len, device):
    """
    Creates a binary mask for valid sequence lengths.

    Arguments
    ---------
    lengths : torch.Tensor
        A 1D tensor of integers representing the lengths of each sequence
        in the batch. e.g., [100, 80, 120]
    max_len : int
        The maximum length to which the mask should be created.
    device : torch.device
        The device to create the tensors on.

    Returns
    -------
    torch.Tensor
        A binary mask of shape [Batch_Size, Max_Length].
    """
    # Create a range tensor [0, 1, 2, ..., max_len-1]
    indices = torch.arange(max_len, device=device).unsqueeze(0)
    # Compare lengths to indices.
    mask = indices < lengths.unsqueeze(1)
    return mask


class Conv1dBlock(nn.Module):
    """
    A 1D convolution block composed of Conv1d + Activation + BatchNorm + Dropout.
    - If `kernel_size=1`, this block acts as a frame-wise Feed-Forward Network (FFN).
    - If `kernel_size > 1`, this block acts as a Time-Delay Neural Network (TDNN) layer.

    Arguments
    ---------
    input_channels : int
        Number of input channels.
    output_channels : int
        The number of output channels.
    kernel_size : int
        The kernel size of the 1D convolution.
    dilation : int
        The dilation of the 1D convolution.
    activation : torch class
        A class for constructing the activation layer (e.g., nn.ReLU).
    groups : int
        The groups size of the 1D convolution.
    dropout : float
        Rate of dropout during training.
    """

    def __init__(
        self,
        input_channels,
        output_channels,
        kernel_size,
        dilation=1,
        activation=nn.ReLU,
        groups=1,
        dropout=0.0,
    ):
        super().__init__()

        # Calculate 'same' padding to keep the temporal length constant
        self.padding = dilation * (kernel_size - 1) // 2

        self.conv = nn.Conv1d(
            in_channels=input_channels,
            out_channels=output_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=self.padding,
            groups=groups,
        )
        self.activation = activation()
        self.norm = nn.BatchNorm1d(num_features=output_channels)
        self.dropout = nn.Dropout1d(p=dropout)

    def forward(self, x):
        """
        Processes the input tensor x and returns an output tensor.
        Input shape: [Batch, Channels, Length]
        """
        return self.dropout(self.norm(self.activation(self.conv(x))))


class AttentiveStatisticsPooling(nn.Module):
    """
    This class implements an attentive statistics pooling layer.
    It learns to assign different weights to each frame (time-step) and
    computes the weighted mean for the utterance.

    This replaces a static mean pooling by a learnable weighted average.
    It always uses global context (the utterance-level mean) to help
    determine the frame-level attention weights.

    Arguments
    ---------
    input_channels: int
        The number of input channels (feature dimensions)
        of the frame-level features.
    attention_channels: int
        The number of hidden channels in the attention mechanism FFN.

    Example
    -------
    >>> # Input is [Batch, Time, Channels]
    >>> features = torch.rand([8, 120, 64])
    >>> # Lengths are absolute frame counts
    >>> lengths = torch.randint(80, 120, (8,))
    >>> asp_layer = AttentiveStatisticsPooling(64, attention_channels=128)
    >>> out_tensor = asp_layer(features, lengths)
    >>> # Output is [Batch, Channels]
    >>> out_tensor.shape
    torch.Size([8, 64])
    """

    def __init__(self, input_channels, attention_channels=128):
        super().__init__()

        self.eps = 1e-12
        
        # Input features + global mean
        attn_input_channels = input_channels * 2

        # This is the first "hidden" layer of the attention FFN.
        self.attention_hidden_layer = Conv1dBlock(
            input_channels=attn_input_channels,
            output_channels=attention_channels,
            kernel_size=1,
            activation=nn.ReLU,
        )

        self.tanh = nn.Tanh()

        # This is the second "scoring" layer of the attention FFN.
        self.attention_scoring_layer = nn.Conv1d(
            in_channels=attention_channels,
            out_channels=input_channels,  # Output C scores
            kernel_size=1,
        )

    @staticmethod
    def _compute_weighted_mean(features, weights, dim=2, eps=1e-12):
        """
        Computes the weighted mean.

        Arguments
        ---------
        features : torch.Tensor
            Input tensor of shape [N, C, L].
        weights : torch.Tensor
            Weights tensor of shape [N, C, L] (must sum to 1 over dim=2).
        dim : int
            The dimension to compute statistics over (the time dimension).
        eps : float
            A small value to clamp variance for numerical stability.

        Returns
        -------
        torch.Tensor
            The weighted mean.
        """
        # Weighted mean
        mean = (weights * features).sum(dim)
        return mean

    def forward(self, features, lengths):
        """
        Calculates weighted mean for a batch of features.

        Arguments
        ---------
        features : torch.Tensor
            Tensor of shape [Batch, Time, Channels].
        lengths : torch.Tensor
            A 1D tensor of *absolute* lengths (frame counts) for each
            sequence in the batch.

        Returns
        -------
        pooled_stats : torch.Tensor
            The weighted_mean of shape [Batch, Channels].
        """
        batch_size, max_time, num_channels = features.shape

        # Permute to [Batch, Channels, Time] for Conv1d layers
        features = features.permute(0, 2, 1)

        # 1. Create mask from lengths
        # `lengths` is absolute, so we use it directly
        mask = length_to_mask(
            lengths, max_len=max_time, device=features.device
        )
        mask = mask.unsqueeze(1)  # Shape [N, 1, L] for broadcasting

        # 2. Prepare features for the attention mechanism (with global context)
        # Compute simple (unweighted) global mean
        total = mask.sum(dim=2, keepdim=True).float().clamp(min=self.eps)
        unweighted_mask = mask / total

        # Compute global mean
        global_mean = self._compute_weighted_mean(
            features, unweighted_mask, eps=self.eps
        )

        # Repeat global stats to match the time dimension [N, C, L]
        global_mean = global_mean.unsqueeze(2).repeat(1, 1, max_time)

        # Concatenate global context
        # Shape: [N, C*2, L]
        attn_input = torch.cat([features, global_mean], dim=1)

        # 3. Compute attention scores
        # First layer: [N, C*2, L] -> [N, A, L]
        hidden = self.attention_hidden_layer(attn_input)
        hidden = self.tanh(hidden)

        # Second layer (scoring): [N, A, L] -> [N, C, L]
        scores = self.attention_scoring_layer(hidden)

        # 4. Mask scores and apply softmax
        scores = scores.masked_fill(mask == 0, float("-inf"))
        attention_weights = F.softmax(scores, dim=2)

        # 5. Compute weighted statistics (mean only)
        weighted_mean = self._compute_weighted_mean(
            features, attention_weights, eps=self.eps
        )

        # 6. Return final pooled vector
        # Shape: [N, C]
        return weighted_mean