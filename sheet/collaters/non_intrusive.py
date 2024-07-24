#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2024 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

FEAT_TYPES = ["waveform", "mag_sgram"]


class NonIntrusiveCollater(object):
    """Customized collater for Pytorch DataLoader in the non-intrusive setting."""

    def __init__(self, padding_mode):
        """Initialize customized collater for PyTorch DataLoader."""
        self.padding_mode = padding_mode

    def __call__(self, batch):
        """Convert into batch tensors."""

        items = {}
        sorted_batch = sorted(batch, key=lambda x: -x["waveform"].shape[0])
        bs = len(sorted_batch)  # batch_size
        all_keys = list(sorted_batch[0].keys())

        # score & listener id
        items["scores"] = torch.tensor(
            [sorted_batch[i]["score"] for i in range(bs)], dtype=torch.float
        )
        items["avg_scores"] = torch.tensor(
            [sorted_batch[i]["avg_score"] for i in range(bs)], dtype=torch.float
        )
        if "listener_id" in all_keys:
            items["listener_ids"] = [sorted_batch[i]["listener_id"] for i in range(bs)]
        if "listener_idx" in all_keys:
            items["listener_idxs"] = torch.tensor(
                [sorted_batch[i]["listener_idx"] for i in range(bs)], dtype=torch.long
            )

        # phoneme and reference
        if "phoneme_idxs" in all_keys:
            phonemes = [
                torch.LongTensor(sorted_batch[i]["phoneme_idxs"]) for i in range(bs)
            ]
            items["phoneme_lengths"] = torch.from_numpy(
                np.array([phoneme.size(0) for phoneme in phonemes])
            )
            items["phoneme_idxs"] = pad_sequence(phonemes, batch_first=True)
        if "reference_idxs" in all_keys:
            references = [
                torch.LongTensor(sorted_batch[i]["reference_idxs"]) for i in range(bs)
            ]
            items["reference_lengths"] = torch.from_numpy(
                np.array([reference.size(0) for reference in references])
            )
            items["reference_idxs"] = pad_sequence(references, batch_first=True)

        # ids
        items["system_ids"] = [sorted_batch[i]["system_id"] for i in range(bs)]
        items["sample_ids"] = [sorted_batch[i]["sample_id"] for i in range(bs)]

        # pad input features (only those in FEAT TYPES)
        for feat_type in FEAT_TYPES:
            if not feat_type in sorted_batch[0]:
                continue

            feats = [sorted_batch[i][feat_type] for i in range(bs)]
            feat_lengths = torch.from_numpy(np.array([feat.size(0) for feat in feats]))

            # padding
            if self.padding_mode == "zero_padding":
                feats_padded = pad_sequence(feats, batch_first=True)
            elif self.padding_mode == "repetitive":
                max_len = feat_lengths[0]
                feats_padded = []
                for feat in feats:
                    this_len = feat.shape[0]
                    dup_times = max_len // this_len
                    remain = max_len - this_len * dup_times
                    to_dup = [feat for t in range(dup_times)]
                    to_dup.append(feat[:remain])
                    duplicated_feat = torch.Tensor(np.concatenate(to_dup, axis=0))
                    feats_padded.append(duplicated_feat)
                feats_padded = torch.stack(feats_padded, dim=0)
            else:
                raise NotImplementedError

            items[feat_type] = feats_padded
            items[feat_type + "_lengths"] = feat_lengths

        return items
