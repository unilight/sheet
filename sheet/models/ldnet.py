# -*- coding: utf-8 -*-

# Copyright 2024 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

# LDNet model
# taken from: https://github.com/unilight/LDNet/blob/main/models/LDNet.py (written by myself)

import math

import torch
import torch.nn as nn

from sheet.modules.ldnet.modules import STRIDE, MobileNetV3ConvBlocks, Projection


class LDNet(torch.nn.Module):
    def __init__(
        self,
        model_input: str,
        # listener related
        num_listeners: int,
        listener_emb_dim: int,
        use_mean_listener: bool,
        # model related
        activation: str,
        encoder_type: str,
        encoder_bneck_configs: list,
        encoder_output_dim: int,
        decoder_type: str,
        decoder_dnn_dim: int,
        output_type: str,
        range_clipping: bool,
        # mean net related
        use_mean_net: bool = False,
        mean_net_type: str = "ffn",
        mean_net_dnn_dim: int = 64,
        mean_net_range_clipping: bool = True,
    ):
        super().__init__()  # this is needed! or else there will be an error.
        self.use_mean_listener = use_mean_listener
        self.output_type = output_type

        # only accept mag_sgram as input
        assert model_input == "mag_sgram"

        # define listener embedding
        self.num_listeners = num_listeners
        self.listener_embeddings = nn.Embedding(
            num_embeddings=num_listeners, embedding_dim=listener_emb_dim
        )

        # define activation
        if activation == "ReLU":
            self.activation = nn.ReLU
        else:
            raise NotImplementedError

        # define encoder
        if encoder_type == "mobilenetv3":
            self.encoder = MobileNetV3ConvBlocks(
                encoder_bneck_configs, encoder_output_dim
            )
        else:
            raise NotImplementedError

        # define decoder
        self.decoder_type = decoder_type
        if decoder_type == "ffn":
            decoder_dnn_input_dim = encoder_output_dim + listener_emb_dim
        else:
            raise NotImplementedError
        # there is always dnn
        self.decoder_dnn = Projection(
            decoder_dnn_input_dim,
            decoder_dnn_dim,
            self.activation,
            output_type,
            range_clipping,
        )

        # define mean net
        self.use_mean_net = use_mean_net
        self.mean_net_type = mean_net_type
        if use_mean_net:
            if mean_net_type == "ffn":
                mean_net_dnn_input_dim = encoder_output_dim
            else:
                raise NotImplementedError
            # there is always dnn
            self.mean_net_dnn = Projection(
                mean_net_dnn_input_dim,
                mean_net_dnn_dim,
                self.activation,
                output_type,
                mean_net_range_clipping,
            )

    def _get_output_dim(self, input_size, num_layers, stride=STRIDE):
        """
        calculate the final ouptut width (dim) of a CNN using the following formula
        w_i = |_ (w_i-1 - 1) / stride + 1 _|
        """
        output_dim = input_size
        for _ in range(num_layers):
            output_dim = math.floor((output_dim - 1) / STRIDE + 1)
        return output_dim

    def get_num_params(self):
        return sum(p.numel() for n, p in self.named_parameters())

    def forward(self, inputs):
        """Calculate forward propagation.
        Args:
            mag_sgram has shape (batch, time, dim)
            listener_ids has shape (batch)
        """
        mag_sgram = inputs["mag_sgram"]
        mag_sgram_lengths = inputs["mag_sgram_lengths"]
        listener_ids  = inputs["listener_idxs"]

        batch, time, _ = mag_sgram.shape

        # get listener embedding
        listener_embs = self.listener_embeddings(listener_ids)  # (batch, emb_dim)
        listener_embs = torch.stack(
            [listener_embs for i in range(time)], dim=1
        )  # (batch, time, feat_dim)

        # encoder and inject listener embedding
        mag_sgram = mag_sgram.unsqueeze(1)
        encoder_outputs = self.encoder(mag_sgram)  # (batch, ch, time, feat_dim)
        encoder_outputs = encoder_outputs.view(
            (batch, time, -1)
        )  # (batch, time, feat_dim)
        decoder_inputs = torch.cat(
            [encoder_outputs, listener_embs], dim=-1
        )  # concat along feature dimension

        # mean net
        if self.use_mean_net:
            mean_net_inputs = encoder_outputs
            if self.mean_net_type == "rnn":
                mean_net_outputs, (h, c) = self.mean_net_rnn(mean_net_inputs)
            else:
                mean_net_outputs = mean_net_inputs
            mean_net_outputs = self.mean_net_dnn(
                mean_net_outputs
            )  # [batch, time, 1 (scalar) / 5 (categorical)]

        # decoder
        if self.decoder_type == "rnn":
            decoder_outputs, (h, c) = self.decoder_rnn(decoder_inputs)
        else:
            decoder_outputs = decoder_inputs
        decoder_outputs = self.decoder_dnn(
            decoder_outputs
        )  # [batch, time, 1 (scalar) / 5 (categorical)]

        # define scores
        ret = {
            "frame_lengths": mag_sgram_lengths,
            "mean_scores": mean_net_outputs if self.use_mean_net else None,
            "ld_scores": decoder_outputs
        }
        

        return ret

    def mean_listener_inference(self, inputs):
        """Mean listener inference.
        Args:
            mag_sgram has shape (batch, time, dim)
        """

        assert self.use_mean_listener
        mag_sgram = inputs["mag_sgram"]
        batch, time, dim = mag_sgram.shape
        device = mag_sgram.device

        # get listener embedding
        listener_id = (torch.ones(batch, dtype=torch.long) * self.num_listeners - 1).to(
            device
        )  # (bs)
        listener_embs = self.listener_embeddings(listener_id)  # (bs, emb_dim)
        listener_embs = torch.stack(
            [listener_embs for i in range(time)], dim=1
        )  # (batch, time, feat_dim)

        # encoder and inject listener embedding
        mag_sgram = mag_sgram.unsqueeze(1)
        encoder_outputs = self.encoder(mag_sgram)  # (batch, ch, time, feat_dim)
        encoder_outputs = encoder_outputs.view(
            (batch, time, -1)
        )  # (batch, time, feat_dim)
        decoder_inputs = torch.cat(
            [encoder_outputs, listener_embs], dim=-1
        )  # concat along feature dimension

        # decoder
        if self.decoder_type == "rnn":
            decoder_outputs, (h, c) = self.decoder_rnn(decoder_inputs)
        else:
            decoder_outputs = decoder_inputs
        decoder_outputs = self.decoder_dnn(
            decoder_outputs
        )  # [batch, time, 1 (scalar) / 5 (categorical)]

        # define scores
        decoder_outputs = decoder_outputs.squeeze(-1)
        scores = torch.mean(decoder_outputs, dim=1)
        return {"scores": scores}

    def average_inference(self, mag_sgram, include_meanspk=False):
        """Average listener inference.
        Args:
            mag_sgram has shape (batch, time, dim)
        """

        bs, time, _ = mag_sgram.shape
        device = mag_sgram.device
        if self.use_mean_listener and not include_meanspk:
            actual_num_listeners = self.num_listeners - 1
        else:
            actual_num_listeners = self.num_listeners

        # all listener ids
        listener_id = (
            torch.arange(actual_num_listeners, dtype=torch.long)
            .repeat(bs, 1)
            .to(device)
        )  # (bs, nj)
        listener_embs = self.listener_embedding(listener_id)  # (bs, nj, emb_dim)
        listener_embs = torch.stack(
            [listener_embs for i in range(time)], dim=2
        )  # (bs, nj, time, feat_dim)

        # encoder and inject listener embedding
        mag_sgram = mag_sgram.unsqueeze(1)
        encoder_outputs = self.encoder(mag_sgram)  # (batch, ch, time, feat_dim)
        encoder_outputs = encoder_outputs.view(
            (bs, time, -1)
        )  # (batch, time, feat_dim)
        decoder_inputs = torch.stack(
            [encoder_outputs for i in range(actual_num_listeners)], dim=1
        )  # (bs, nj, time, feat_dim)
        decoder_inputs = torch.cat(
            [decoder_inputs, listener_embs], dim=-1
        )  # concat along feature dimension

        # mean net
        if self.use_mean_net:
            mean_net_inputs = encoder_outputs
            if self.mean_net_type == "rnn":
                mean_net_outputs, (h, c) = self.mean_net_rnn(mean_net_inputs)
            else:
                mean_net_outputs = mean_net_inputs
            mean_net_outputs = self.mean_net_dnn(
                mean_net_outputs
            )  # [batch, time, 1 (scalar) / 5 (categorical)]

        # decoder
        if self.decoder_type == "rnn":
            decoder_outputs = decoder_inputs.view((bs * actual_num_listeners, time, -1))
            decoder_outputs, (h, c) = self.decoder_rnn(decoder_outputs)
        else:
            decoder_outputs = decoder_inputs
        decoder_outputs = self.decoder_dnn(
            decoder_outputs
        )  # [batch, time, 1 (scalar) / 5 (categorical)]
        decoder_outputs = decoder_outputs.view(
            (bs, actual_num_listeners, time, -1)
        )  # (bs, nj, time, 1/5)

        if self.output_type == "scalar":
            decoder_outputs = decoder_outputs.squeeze(-1)  # (bs, nj, time)
            posterior_scores = torch.mean(decoder_outputs, dim=2)
            ld_scores = torch.mean(decoder_outputs, dim=1)  # (bs, time)
        elif self.output_type == "categorical":
            ld_posterior = torch.nn.functional.softmax(decoder_outputs, dim=-1)
            ld_scores = torch.inner(
                ld_posterior, torch.Tensor([1, 2, 3, 4, 5]).to(device)
            )
            posterior_scores = torch.mean(ld_scores, dim=2)
            ld_scores = torch.mean(ld_scores, dim=1)  # (bs, time)

        # define scores
        scores = torch.mean(ld_scores, dim=1)

        return {"scores": scores, "posterior_scores": posterior_scores}
