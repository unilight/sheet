# -*- coding: utf-8 -*-

# Copyright 2024 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

"""Non-intrusive dataset modules."""

from collections import defaultdict
from multiprocessing import Manager

import numpy as np
import torch.nn.functional as F
import torchaudio
from sheet.utils import read_csv
from torch.utils.data import Dataset

MIN_REQUIRED_WAV_LENGTH = 1040


class NonIntrusiveDataset(Dataset):
    """PyTorch compatible mel-to-mel dataset for parallel VC."""

    def __init__(
        self,
        csv_path,
        target_sample_rate,
        model_input="wav",
        wav_only=False,
        use_mean_listener=False,
        use_phoneme=False,
        symbols=None,
        categorical=False,
        categorical_step=1.0,
        no_feat=False,
        allow_cache=False,
    ):
        """Initialize dataset.

        Args:
            csv path (str): path to the csv file
            target_sample_rate (int): resample to this seample rate if there is a mismatch.
            model_input (str): defalut is wav. is this is mag_sgram, extract magnitute sgram.
            wav_only (bool): whether to return only wavs. Basically this means inference mode.
            use_mean_listener (bool): whether to use mean listener. (only for datasets with listener labels)
            use_phoneme (bool): whether to use phoneme. (only for UTMOS training)
            symbols (str): symbols for phoneme. (only for UTMOS training)
            categorical (bool): whether to use categorical output.
            categorical_step (float): step for the categorical output. defauly is 1.0.
            no_feat (bool): Whether to skip loading features (waveforms, mag_sgrams ...)
            allow_cache (bool): Whether to allow cache of the loaded files.

        """
        self.target_sample_rate = target_sample_rate
        self.use_phoneme = use_phoneme
        if self.use_phoneme:
            self.symbols = symbols
        self.resamplers = {}
        assert csv_path != ""
        self.categorical = categorical
        self.categorical_step = categorical_step
        self.no_feat = no_feat

        # set model input transform
        self.model_input = model_input
        if model_input == "mag_sgram":
            self.mag_sgram_transform = torchaudio.transforms.Spectrogram(
                n_fft=512, hop_length=256, win_length=512, power=1
            )

        # read csv file
        self.metadata, _ = read_csv(csv_path, dict_reader=True)

        # calculate average score for each sample and add to metadata
        self.calculate_avg_score()

        if wav_only:
            self.reduce_to_wav_only()
        else:
            # add mean listener to metadata
            if use_mean_listener:
                mean_listener_metadata = self.gen_mean_listener_metadata()
                self.metadata = self.metadata + mean_listener_metadata

            # get num of listeners
            self.num_listeners = self.get_num_listeners()

        # get num of domains if domain_idx exists
        if "domain_idx" in self.metadata[0]:
            self.num_domains = self.get_num_domains()

        # build hash
        self.build_feat_hash()

        # set cache
        self.allow_cache = allow_cache
        if allow_cache:
            # NOTE(kan-bayashi): Manager is need to share memory in dataloader with num_workers > 0
            self.manager = Manager()
            self.wav_caches = self.manager.list()
            self.wav_caches += [() for _ in range(self.num_wavs)]
            if self.model_input == "mag_sgram":
                self.mag_sgram_caches = self.manager.list()
                self.mag_sgram_caches += [() for _ in range(self.num_wavs)]

    def __len__(self):
        """Return dataset length.

        Returns:
            int: The length of dataset.

        """
        return len(self.metadata)

    def get_num_listeners(self):
        """Get number of listeners by counting unique listener id"""
        listener_ids = set()
        for item in self.metadata:
            listener_ids.add(item["listener_id"])
        return len(listener_ids)

    def get_num_domains(self):
        """Get number of domains by counting unique domain idxs"""
        domain_idxs = set()
        for item in self.metadata:
            domain_idxs.add(item["domain_idx"])
        return len(domain_idxs)

    def build_feat_hash(self):
        sample_ids = {}
        count = 0
        for i in range(len(self.metadata)):
            item = self.metadata[i]
            sample_id = item["sample_id"]
            if not sample_id in sample_ids:
                sample_ids[sample_id] = count
                count += 1
            self.metadata[i]["hash_id"] = sample_ids[sample_id]
        self.num_wavs = len(sample_ids)

    def __getitem__(self, idx):
        item = self.metadata[idx]

        # handle score
        item["score"] = float(item["score"])  # cast from str to int
        if self.categorical:
            # we assume the score always starts from 1
            item["score"] = int((item["score"] - 1) // self.categorical_step)

        if "listener_idx" in item:
            item["listener_idx"] = int(item["listener_idx"])  # cast from str to int
        if "domain_idx" in item:
            item["domain_idx"] = int(item["domain_idx"])  # cast from str to int
        hash_id = item["hash_id"]

        # process text
        if self.use_phoneme:
            if "phoneme" in item:
                if "phoneme_idxs" not in item:
                    item["phoneme_idxs"] = [
                        self.symbols.index(p) for p in item["phoneme"]
                    ]
            if "reference" in item:
                if "reference_idxs" not in item:
                    item["reference_idxs"] = [
                        self.symbols.index(p) for p in item["reference"]
                    ]

        # fetch waveform. return cached item if exists
        if not self.no_feat:
            if self.allow_cache and len(self.wav_caches[hash_id]) != 0:
                item["waveform"] = self.wav_caches[hash_id]
            else:
                # read waveform
                waveform, sample_rate = torchaudio.load(
                    item["wav_path"], channels_first=False
                )  # waveform: [T, 1]
                # resample if needed
                if sample_rate != self.target_sample_rate:
                    resampler_key = f"{sample_rate}-{self.target_sample_rate}"
                    if resampler_key not in self.resamplers:
                        self.resamplers[resampler_key] = torchaudio.transforms.Resample(
                            sample_rate, self.target_sample_rate, dtype=waveform.dtype
                        )
                    waveform = self.resamplers[resampler_key](waveform)

                waveform = waveform.squeeze(-1)

                # always pad to a minumum length
                if waveform.shape[0] < MIN_REQUIRED_WAV_LENGTH:
                    to_pad = (MIN_REQUIRED_WAV_LENGTH - waveform.shape[0]) // 2
                    waveform = F.pad(waveform, (to_pad, to_pad), "constant", 0)

                item["waveform"] = waveform
                if self.allow_cache:
                    self.wav_caches[hash_id] = item["waveform"]

        # additional feature extraction
        if not self.no_feat:
            if self.model_input == "mag_sgram":
                # fetch mag_sgram. return cached item if exists
                if self.allow_cache and len(self.mag_sgram_caches[hash_id]) != 0:
                    item["mag_sgram"] = self.mag_sgram_caches[hash_id]
                else:
                    # torchaudio requires waveform to be [..., T]
                    mag_sgram = self.mag_sgram_transform(
                        waveform.squeeze(-1)
                    )  # mag_sgram: [freq, T]
                    item["mag_sgram"] = mag_sgram.mT  # [T, freq]
                    if self.allow_cache:
                        self.mag_sgram_caches[hash_id] = item["mag_sgram"]

        return item

    def calculate_avg_score(self):
        sample_scores = defaultdict(list)

        # loop through metadata
        for item in self.metadata:
            sample_scores[item["sample_id"]].append(float(item["score"]))

        # take average
        sample_avg_score = {
            sample_id: np.mean(np.array(scores))
            for sample_id, scores in sample_scores.items()
        }
        self.sample_avg_score = sample_avg_score

        # fill back into metadata
        for i, item in enumerate(self.metadata):
            self.metadata[i]["avg_score"] = sample_avg_score[item["sample_id"]]
            if self.categorical:
                # we assume the score always starts from 1
                self.metadata[i]["avg_score"] = int(
                    (self.metadata[i]["avg_score"] - 1) // self.categorical_step
                )

    def gen_mean_listener_metadata(self):
        mean_listener_metadata = []
        sample_ids = set()
        for item in self.metadata:
            sample_id = item["sample_id"]
            if sample_id not in sample_ids:
                new_item = {k: v for k, v in item.items()}
                new_item["listener_id"] = "mean_listener"
                mean_listener_metadata.append(new_item)
                sample_ids.add(sample_id)
        return mean_listener_metadata

    def reduce_to_wav_only(self):
        new_metadata = {}  # {sample_id: item}
        for item in self.metadata:
            sample_id = item["sample_id"]
            if not sample_id in new_metadata:
                new_metadata[sample_id] = {
                    k: v
                    for k, v in item.items()
                    if k not in ["listener_id", "listener_idx"]
                }

        self.metadata = list(new_metadata.values())
