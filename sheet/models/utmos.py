# -*- coding: utf-8 -*-

# Copyright 2024 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

# UTMOS model
# modified from: https://github.com/sarulab-speech/UTMOS22/tree/master/strong

import math

import torch
import torch.nn as nn

from sheet.modules.ldnet.modules import Projection
from sheet.modules.utils import make_non_pad_mask


class UTMOS(torch.nn.Module):
    def __init__(
        self,
        model_input: str,
        # model related
        ssl_module: str,
        s3prl_name: str,
        ssl_model_output_dim: int,
        ssl_model_layer_idx: int,
        # phoneme and reference related
        use_phoneme: bool = True,
        phoneme_encoder_dim: int = 256,
        phoneme_encoder_emb_dim: int = 256,
        phoneme_encoder_out_dim: int = 256,
        phoneme_encoder_n_lstm_layers: int = 3,
        phoneme_encoder_vocab_size: int = 300,
        use_reference: bool = True,
        # listener related
        use_listener_modeling: bool = False,
        num_listeners: int = None,
        listener_emb_dim: int = None,
        use_mean_listener: bool = True,
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
        decoder_input_dim = ssl_model_output_dim
        
        # define phoneme encoder
        self.use_phoneme = use_phoneme
        self.use_reference = use_reference
        if self.use_phoneme:
            self.phoneme_embedding = nn.Embedding(phoneme_encoder_vocab_size, phoneme_encoder_emb_dim)
            self.phoneme_encoder_lstm = nn.LSTM(phoneme_encoder_emb_dim, phoneme_encoder_dim,
                                num_layers=phoneme_encoder_n_lstm_layers, dropout=0.1, bidirectional=True)
            if self.use_reference:
                
                phoneme_encoder_linear_input_dim = phoneme_encoder_dim + phoneme_encoder_dim
            else:
                phoneme_encoder_linear_input_dim = phoneme_encoder_dim
            self.phoneme_encoder_linear = nn.Sequential(
                nn.Linear(phoneme_encoder_linear_input_dim, phoneme_encoder_out_dim),
                nn.ReLU()
            )
            decoder_input_dim += phoneme_encoder_out_dim

        # NOTE(unlight): ignore domain embedding right now

        # listener modeling related
        self.use_listener_modeling = use_listener_modeling
        if use_listener_modeling:
            self.num_listeners = num_listeners
            self.listener_embeddings = nn.Embedding(
                num_embeddings=num_listeners, embedding_dim=listener_emb_dim
            )
            decoder_input_dim += listener_emb_dim

        # define decoder rnn
        self.use_decoder_rnn = use_decoder_rnn
        if self.use_decoder_rnn:
            self.decoder_rnn = nn.LSTM(
                input_size = decoder_input_dim,
                hidden_size = decoder_rnn_dim,
                num_layers = 1,
                batch_first = True,
                bidirectional = True
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
        """
        waveform, waveform_lengths = inputs["waveform"], inputs["waveform_lengths"]

        # ssl model forward
        ssl_model_outputs, ssl_model_output_lengths = self.ssl_model_forward(waveform, waveform_lengths)
        to_concat = [ssl_model_outputs]
        time = ssl_model_outputs.size(1)

        # phoneme encoder forward
        if self.use_phoneme:
            phoneme_encoder_outputs = self.phoneme_encoder_forward(inputs, time)
            to_concat.append(phoneme_encoder_outputs)

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
            "frame_lengths": ssl_model_output_lengths
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
        ssl_model_outputs, ssl_model_output_lengths = self.ssl_model_forward(waveform, waveform_lengths)
        to_concat = [ssl_model_outputs]
        time = ssl_model_outputs.size(1)

        # phoneme encoder forward
        if self.use_phoneme:
            phoneme_encoder_outputs = self.phoneme_encoder_forward(inputs, time)
            to_concat.append(phoneme_encoder_outputs)

        # get listener embedding
        if self.use_listener_modeling:
            device = waveform.device
            listener_ids = (torch.ones(batch, dtype=torch.long) * self.num_listeners - 1).to(
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
        ssl_model_output_lengths = all_ssl_model_output_lengths[self.ssl_model_layer_idx]
        return ssl_model_outputs, ssl_model_output_lengths
    
    def phoneme_encoder_forward(self, inputs, time):
        phoneme, phoneme_lengths = inputs["phoneme_idxs"], inputs["phoneme_lengths"]
        phoneme_embeddings = self.phoneme_embedding(phoneme)
        phoneme_embeddings = torch.nn.utils.rnn.pack_padded_sequence(
            phoneme_embeddings, phoneme_lengths, batch_first=True, enforce_sorted=False)
        _, (phoneme_encoder_outputs, _) = self.phoneme_encoder_lstm(phoneme_embeddings)
        phoneme_encoder_outputs = phoneme_encoder_outputs[-1] + phoneme_encoder_outputs[0]
        if self.use_reference:
            assert "reference_idxs" in inputs and "reference_lengths" in inputs, "reference and reference_lenghts should not be None when use_reference is True"
            reference, reference_lengths = inputs["reference_idxs"], inputs["reference_lengths"]
            reference_embeddings = self.phoneme_embedding(reference)
            reference_embeddings = torch.nn.utils.rnn.pack_padded_sequence(
                reference_embeddings, reference_lengths, batch_first=True, enforce_sorted=False)
            _, (reference_encoder_outputs, _) = self.phoneme_encoder_lstm(reference_embeddings)
            reference_encoder_outputs = reference_encoder_outputs[-1] + reference_encoder_outputs[0]
            phoneme_encoder_outputs = self.phoneme_encoder_linear(torch.cat([phoneme_encoder_outputs, reference_encoder_outputs],1))
        else:
            phoneme_encoder_outputs = self.phoneme_encoder_linear(phoneme_encoder_outputs)

        # expand
        phoneme_encoder_outputs = torch.stack(
            [phoneme_encoder_outputs for i in range(time)], dim=1
        )  # (batch, time, feat_dim)

        return phoneme_encoder_outputs