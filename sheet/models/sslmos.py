# -*- coding: utf-8 -*-

# Copyright 2024 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

# SSLMOS model
# modified from: https://github.com/nii-yamagishilab/mos-finetune-ssl/blob/main/mos_fairseq.py (written by Erica Cooper)

import math

import torch
import torch.nn as nn

from sheet.modules.ldnet.modules import Projection
from sheet.modules.utils import make_non_pad_mask


class SSLMOS(torch.nn.Module):
    def __init__(
        self,
        model_input: str,
        # model related
        ssl_module: str,
        s3prl_name: str,
        ssl_model_output_dim: int,
        ssl_model_layer_idx: int,
        # mean net related
        mean_net_dnn_dim: int = 64,
        mean_net_output_type: str = "scalar",
        mean_net_range_clipping: bool = True,
        # listener related
        use_listener_modeling: bool = False,
        num_listeners: int = None,
        listener_emb_dim: int = None,
        use_mean_listener: bool = True,
        decoder_type: str = "ffn",
        decoder_dnn_dim: int = 64,
        output_type: str = "scalar",
        range_clipping: bool = True,
    ):
        super().__init__()  # this is needed! or else there will be an error.
        self.use_mean_listener = use_mean_listener
        self.output_type = output_type

        # define listener embedding
        self.use_listener_modeling = use_listener_modeling

        # define ssl model
        if ssl_module == "s3prl":
            from s3prl.nn import S3PRLUpstream

            if s3prl_name in S3PRLUpstream.available_names():
                self.ssl_model = S3PRLUpstream(s3prl_name)
            self.ssl_model_layer_idx = ssl_model_layer_idx
        else:
            raise NotImplementedError

        # default uses ffn type mean net
        self.mean_net_dnn = Projection(
            ssl_model_output_dim,
            mean_net_dnn_dim,
            nn.ReLU,
            mean_net_output_type,
            mean_net_range_clipping,
        )

        # listener modeling related
        self.use_listener_modeling = use_listener_modeling
        if use_listener_modeling:
            self.num_listeners = num_listeners
            self.listener_embeddings = nn.Embedding(
                num_embeddings=num_listeners, embedding_dim=listener_emb_dim
            )
            # define decoder
            self.decoder_type = decoder_type
            if decoder_type == "ffn":
                decoder_dnn_input_dim = ssl_model_output_dim + listener_emb_dim
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

    def get_num_params(self):
        return sum(p.numel() for n, p in self.named_parameters())

    def forward(self, inputs):
        """Calculate forward propagation.
        Args:
            waveform has shape (batch, time)
            waveform_lengths has shape (batch)
            listener_ids has shape (batch)
        """
        waveform = inputs["waveform"]
        waveform_lengths = inputs["waveform_lengths"]

        batch, time = waveform.shape

        # get listener embedding
        if self.use_listener_modeling:
            listener_ids = inputs["listener_idxs"]
            # NOTE(unlight): not tested yet
            listener_embs = self.listener_embeddings(listener_ids)  # (batch, emb_dim)
            listener_embs = torch.stack(
                [listener_embs for i in range(time)], dim=1
            )  # (batch, time, feat_dim)

        # ssl model forward
        all_encoder_outputs, all_encoder_outputs_lens = self.ssl_model(
            waveform, waveform_lengths
        )
        encoder_outputs = all_encoder_outputs[self.ssl_model_layer_idx]
        encoder_outputs_lens = all_encoder_outputs_lens[self.ssl_model_layer_idx]

        # inject listener embedding
        if self.use_listener_modeling:
            # NOTE(unlight): not tested yet
            encoder_outputs = encoder_outputs.view(
                (batch, time, -1)
            )  # (batch, time, feat_dim)
            decoder_inputs = torch.cat(
                [encoder_outputs, listener_embs], dim=-1
            )  # concat along feature dimension
        else:
            decoder_inputs = encoder_outputs

        # masked mean pooling
        # masks = make_non_pad_mask(encoder_outputs_lens)
        # masks = masks.unsqueeze(-1).to(decoder_inputs.device) # [B, max_time, 1]
        # decoder_inputs = torch.sum(decoder_inputs * masks, dim=1) / encoder_outputs_lens.unsqueeze(-1)

        # mean net
        mean_net_outputs = self.mean_net_dnn(
            decoder_inputs
        )  # [batch, time, 1 (scalar) / 5 (categorical)]

        # decoder
        if self.use_listener_modeling:
            if self.decoder_type == "rnn":
                decoder_outputs, (h, c) = self.decoder_rnn(decoder_inputs)
            else:
                decoder_outputs = decoder_inputs
            decoder_outputs = self.decoder_dnn(
                decoder_outputs
            )  # [batch, time, 1 (scalar) / 5 (categorical)]

        # set outputs
        # return lengths for masked loss calculation
        ret = {
            "waveform_lengths": waveform_lengths,
            "frame_lengths": encoder_outputs_lens
        }

        # define scores
        ret["mean_scores"] = mean_net_outputs
        ret["ld_scores"] = decoder_outputs if self.use_listener_modeling else None

        return ret

    def mean_net_inference(self, inputs):
        waveform = inputs["waveform"]
        waveform_lengths = inputs["waveform_lengths"]

        batch, time = waveform.shape

        # ssl model forward
        all_encoder_outputs, all_encoder_outputs_lens = self.ssl_model(
            waveform, waveform_lengths
        )
        encoder_outputs = all_encoder_outputs[self.ssl_model_layer_idx]
        encoder_outputs_lens = all_encoder_outputs_lens[self.ssl_model_layer_idx]

        # mean net
        decoder_inputs = encoder_outputs
        mean_net_outputs = self.mean_net_dnn(
            decoder_inputs
        )  # [batch, time, 1 (scalar) / 5 (categorical)]
        mean_net_outputs = mean_net_outputs.squeeze(-1)
        scores = torch.mean(mean_net_outputs, dim=1)

        return {"scores": scores}
    
    def mean_net_inference_p1(self, waveform, waveform_lengths):
        # ssl model forward
        all_encoder_outputs, _ = self.ssl_model(
            waveform, waveform_lengths
        )
        encoder_outputs = all_encoder_outputs[self.ssl_model_layer_idx]
        return encoder_outputs
    
    def mean_net_inference_p2(self, encoder_outputs):
        # mean net
        mean_net_outputs = self.mean_net_dnn(
            encoder_outputs
        )  # [batch, time, 1 (scalar) / 5 (categorical)]
        mean_net_outputs = mean_net_outputs.squeeze(-1)
        scores = torch.mean(mean_net_outputs, dim=1)

        return scores
