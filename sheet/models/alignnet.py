# -*- coding: utf-8 -*-

# Copyright 2024 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

# Modified AlignNet model

import torch
import torch.nn as nn
from sheet.modules.ldnet.modules import Projection


class AlignNet(torch.nn.Module):
    def __init__(
        self,
        # model related
        ssl_module: str,
        s3prl_name: str,
        ssl_model_output_dim: int,
        ssl_model_layer_idx: int,
        # listener related
        use_listener_modeling: bool = False,
        num_listeners: int = None,
        listener_emb_dim: int = None,
        use_mean_listener: bool = True,
        # domain related
        use_domain_modeling: bool = False,
        num_domains: int = None,
        domain_emb_dim: int = None,
        # decoder related
        use_decoder_rnn: bool = True,
        decoder_rnn_dim: int = 512,
        decoder_dnn_dim: int = 2048,
        decoder_activation: str = "ReLU",
        output_type: str = "scalar",
        range_clipping: bool = True,
    ):
        super().__init__()  # this is needed! or else there will be an error.
        self.use_mean_listener = use_mean_listener
        self.output_type = output_type

        # define ssl model
        if ssl_module == "s3prl":
            from s3prl.nn import S3PRLUpstream

            if s3prl_name in S3PRLUpstream.available_names():
                self.ssl_model = S3PRLUpstream(s3prl_name)
            self.ssl_model_layer_idx = ssl_model_layer_idx
        else:
            raise NotImplementedError
        decoder_input_dim = ssl_model_output_dim

        # listener modeling related
        self.use_listener_modeling = use_listener_modeling
        if use_listener_modeling:
            self.num_listeners = num_listeners
            self.listener_embeddings = nn.Embedding(
                num_embeddings=num_listeners, embedding_dim=listener_emb_dim
            )
            decoder_input_dim += listener_emb_dim

        # domain modeling related
        self.use_domain_modeling = use_domain_modeling
        if use_domain_modeling:
            self.num_domains = num_domains
            self.domain_embeddings = nn.Embedding(
                num_embeddings=num_domains, embedding_dim=domain_emb_dim
            )
            decoder_input_dim += domain_emb_dim

        # define decoder rnn
        self.use_decoder_rnn = use_decoder_rnn
        if self.use_decoder_rnn:
            self.decoder_rnn = nn.LSTM(
                input_size=decoder_input_dim,
                hidden_size=decoder_rnn_dim,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
            )
            decoder_dnn_input_dim = decoder_rnn_dim * 2
        else:
            decoder_dnn_input_dim = decoder_input_dim

        # define activation
        if decoder_activation == "ReLU":
            self.decoder_activation = nn.ReLU
        else:
            raise NotImplementedError

        # there is always decoder dnn
        self.decoder_dnn = Projection(
            decoder_dnn_input_dim,
            decoder_dnn_dim,
            self.decoder_activation,
            output_type,
            range_clipping,
        )

    def get_num_params(self):
        return sum(p.numel() for n, p in self.named_parameters())

    def forward(self, inputs):
        """Calculate forward propagation.
        Args:
            inputs: dict, which has the following keys:
                - waveform has shape (batch, time)
                - waveform_lengths has shape (batch)
                - listener_ids has shape (batch)
                - domain_ids has shape (batch)
        """
        waveform, waveform_lengths = inputs["waveform"], inputs["waveform_lengths"]

        # ssl model forward
        ssl_model_outputs, ssl_model_output_lengths = self.ssl_model_forward(
            waveform, waveform_lengths
        )
        to_concat = [ssl_model_outputs]
        time = ssl_model_outputs.size(1)

        # get listener embedding
        if self.use_listener_modeling:
            listener_ids = inputs["listener_idxs"]
            listener_embs = self.listener_embeddings(listener_ids)  # (batch, emb_dim)
            listener_embs = torch.stack(
                [listener_embs for i in range(time)], dim=1
            )  # (batch, time, feat_dim)

            # NOTE(unilight): is this needed?
            # encoder_outputs = encoder_outputs.view(
            # (batch, time, -1)
            # )  # (batch, time, feat_dim)
            to_concat.append(listener_embs)

        # get domain embedding
        if self.use_domain_modeling:
            domain_ids = inputs["domain_idxs"]
            domain_embs = self.domain_embeddings(domain_ids)  # (batch, emb_dim)
            domain_embs = torch.stack(
                [domain_embs for i in range(time)], dim=1
            )  # (batch, time, feat_dim)

            # NOTE(unilight): is this needed?
            # encoder_outputs = encoder_outputs.view(
            # (batch, time, -1)
            # )  # (batch, time, feat_dim)
            to_concat.append(domain_embs)

        decoder_inputs = torch.cat(to_concat, dim=2)

        # decoder rnn
        if self.use_decoder_rnn:
            decoder_inputs, (h, c) = self.decoder_rnn(decoder_inputs)

        # decoder dnn
        decoder_outputs = self.decoder_dnn(
            decoder_inputs
        )  # [batch, time, 1 (scalar) / 5 (categorical)]

        # set outputs
        # return lengths for masked loss calculation
        ret = {
            "waveform_lengths": waveform_lengths,
            "frame_lengths": ssl_model_output_lengths,
        }
        if self.use_listener_modeling:
            ret["ld_scores"] = decoder_outputs
        else:
            ret["mean_scores"] = decoder_outputs

        return ret

    def mean_listener_inference(self, inputs):
        waveform, waveform_lengths = inputs["waveform"], inputs["waveform_lengths"]
        batch = waveform.size(0)

        # ssl model forward
        ssl_model_outputs, ssl_model_output_lengths = self.ssl_model_forward(
            waveform, waveform_lengths
        )
        to_concat = [ssl_model_outputs]
        time = ssl_model_outputs.size(1)

        # get listener embedding
        if self.use_listener_modeling:
            device = waveform.device
            listener_ids = (
                torch.ones(batch, dtype=torch.long) * self.num_listeners - 1
            ).to(
                device
            )  # (bs)
            listener_embs = self.listener_embeddings(listener_ids)  # (batch, emb_dim)
            listener_embs = torch.stack(
                [listener_embs for i in range(time)], dim=1
            )  # (batch, time, feat_dim)

            # NOTE(unilight): is this needed?
            # encoder_outputs = encoder_outputs.view(
            # (batch, time, -1)
            # )  # (batch, time, feat_dim)
            to_concat.append(listener_embs)

        # get domain embedding
        if self.use_domain_modeling:
            device = waveform.device
            assert "domain_idxs" in inputs, "Must specify domain ID even in inference."
            domain_ids = inputs["domain_idxs"]
            domain_embs = self.domain_embeddings(domain_ids)  # (batch, emb_dim)
            domain_embs = torch.stack(
                [domain_embs for i in range(time)], dim=1
            )  # (batch, time, feat_dim)

            # NOTE(unilight): is this needed?
            # encoder_outputs = encoder_outputs.view(
            # (batch, time, -1)
            # )  # (batch, time, feat_dim)
            to_concat.append(domain_embs)

        decoder_inputs = torch.cat(to_concat, dim=2)

        # decoder rnn
        if self.use_decoder_rnn:
            decoder_inputs, (h, c) = self.decoder_rnn(decoder_inputs)

        # decoder dnn
        decoder_outputs = self.decoder_dnn(
            decoder_inputs
        )  # [batch, time, 1 (scalar) / 5 (categorical)]

        scores = torch.mean(decoder_outputs.squeeze(-1), dim=1)
        return {"scores": scores}

    def ssl_model_forward(self, waveform, waveform_lengths):
        all_ssl_model_outputs, all_ssl_model_output_lengths = self.ssl_model(
            waveform, waveform_lengths
        )
        ssl_model_outputs = all_ssl_model_outputs[self.ssl_model_layer_idx]
        ssl_model_output_lengths = all_ssl_model_output_lengths[
            self.ssl_model_layer_idx
        ]
        return ssl_model_outputs, ssl_model_output_lengths

    def get_ssl_embeddings(self, inputs):
        waveform = inputs["waveform"]
        waveform_lengths = inputs["waveform_lengths"]

        all_encoder_outputs, all_encoder_outputs_lens = self.ssl_model(
            waveform, waveform_lengths
        )
        encoder_outputs = all_encoder_outputs[self.ssl_model_layer_idx]
        return encoder_outputs