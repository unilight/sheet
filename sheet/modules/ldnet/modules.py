# -*- coding: utf-8 -*-

# Copyright 2024 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

# LDNet modules
# taken from: https://github.com/unilight/LDNet/blob/main/models/modules.py (written by myself)

from functools import partial
from typing import List

import torch
from sheet.modules.ldnet.mobilenetv2 import ConvBNActivation
from sheet.modules.ldnet.mobilenetv3 import InvertedResidual as InvertedResidualV3
from sheet.modules.ldnet.mobilenetv3 import InvertedResidualConfig
from torch import nn

STRIDE = 3


class Projection(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim,
        activation,
        output_type,
        _output_dim,
        output_step=1.0,
        range_clipping=False,
    ):
        super(Projection, self).__init__()
        self.output_type = output_type
        self.range_clipping = range_clipping
        if output_type == "scalar":
            output_dim = 1
            if range_clipping:
                self.proj = nn.Tanh()
        elif output_type == "categorical":
            output_dim = _output_dim
            self.output_step = output_step
        else:
            raise NotImplementedError("wrong output_type: {}".format(output_type))

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            activation(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x, inference=False):
        output = self.net(x)

        # scalar / categorical
        if self.output_type == "scalar":
            # range clipping
            if self.range_clipping:
                return self.proj(output) * 2.0 + 3
            else:
                return output
        else:
            if inference:
                return torch.argmax(output, dim=-1) * self.output_step + 1
            else:
                return output


class ProjectionWithUncertainty(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim,
        activation,
        output_type,
        _output_dim,
        output_step=1.0,
        range_clipping=False,
    ):
        super(ProjectionWithUncertainty, self).__init__()
        self.output_type = output_type
        self.range_clipping = range_clipping
        if output_type == "scalar":
            output_dim = 2
            if range_clipping:
                self.proj = nn.Tanh()
        elif output_type == "categorical":
            output_dim = _output_dim
            self.output_step = output_step
        else:
            raise NotImplementedError("wrong output_type: {}".format(output_type))

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            activation(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x, inference=False):
        output = self.net(x) # output shape: [B, T, d]

        # scalar / categorical
        if self.output_type == "scalar":
            mean, logvar = output[:, :, 0], output[:, :, 1]
            # range clipping
            if self.range_clipping:
                return self.proj(mean) * 2.0 + 3, logvar
            else:
                return mean, logvar
        else:
            if inference:
                return torch.argmax(output, dim=-1) * self.output_step + 1
            else:
                return output


class MobileNetV3ConvBlocks(nn.Module):
    def __init__(self, bneck_confs, output_dim):
        super(MobileNetV3ConvBlocks, self).__init__()

        bneck_conf = partial(InvertedResidualConfig, width_mult=1)
        inverted_residual_setting = [bneck_conf(*b_conf) for b_conf in bneck_confs]

        block = InvertedResidualV3

        # Never tested if a different eps and momentum is needed
        # norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)
        norm_layer = nn.BatchNorm2d

        layers: List[nn.Module] = []

        # building first layer
        firstconv_output_channels = inverted_residual_setting[0].input_channels
        layers.append(
            ConvBNActivation(
                1,
                firstconv_output_channels,
                kernel_size=3,
                stride=STRIDE,
                norm_layer=norm_layer,
                activation_layer=nn.Hardswish,
            )
        )

        # building inverted residual blocks
        for cnf in inverted_residual_setting:
            layers.append(block(cnf, norm_layer))

        # building last several layers
        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = output_dim
        layers.append(
            ConvBNActivation(
                lastconv_input_channels,
                lastconv_output_channels,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=nn.Hardswish,
            )
        )
        self.features = nn.Sequential(*layers)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        time = x.shape[2]
        x = self.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, (time, 1))
        x = x.squeeze(-1).transpose(1, 2)
        return x
