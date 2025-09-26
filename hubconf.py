# -*- coding: utf-8 -*-

# Copyright 2025 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

"""torch.hub configuration."""

dependencies = ["torch", "torchaudio", "sheet_sqa"]

import os
import torch
import torch.nn.functional as F
import torchaudio
import yaml

FS = 16000
resamplers = {}
MIN_REQUIRED_WAV_LENGTH = 1040

URLS = {
    "default": {
        "conf": "https://github.com/unilight/sheet/releases/download/v0.1.0/all7-sslmos-mdf-2337-config.yml",
        "model": "https://github.com/unilight/sheet/releases/download/v0.1.0/all7-sslmos-mdf-2337-checkpoint-86000steps.pkl",
    },
    "all8_sslmos_wavlm_large": {
        "conf": "https://github.com/unilight/sheet/releases/download/v0.1.0/all7-sslmos-mdf-2337-config.yml",
        "model": "https://github.com/unilight/sheet/releases/download/v0.1.0/all7-sslmos-mdf-2337-checkpoint-86000steps.pkl",
    },

}

def read_wav(wav_path):
    # read waveform
    waveform, sample_rate = torchaudio.load(
        wav_path, channels_first=False
    )  # waveform: [T, 1]

    # resample if needed
    if sample_rate != FS:
        resampler_key = f"{sample_rate}-{FS}"
        if resampler_key not in resamplers:
            resamplers[resampler_key] = torchaudio.transforms.Resample(
                sample_rate, FS, dtype=waveform.dtype
            )
        waveform = resamplers[resampler_key](waveform)

    waveform = waveform.squeeze(-1)

    # always pad to a minumum length
    if waveform.shape[0] < MIN_REQUIRED_WAV_LENGTH:
        to_pad = (MIN_REQUIRED_WAV_LENGTH - waveform.shape[0]) // 2
        waveform = F.pad(waveform, (to_pad, to_pad), "constant", 0)

    return waveform, sample_rate

class Predictor:
    """Wrapper class for unified waveform reading"""
    def __init__(self, model, config):
        self.model = model
        self.config = config
    
    def predict(self, wav_path=None, wav=None):
        """
        Args:
            wav: must be torch tensor
        """
        if wav is None:
            if wav_path is None:
                raise ValueError("Either wav_path or wav must be set. Please provide one.")
            else:
                wav, _ = read_wav(wav_path)
        else:
            if wav_path is not None:
                raise ValueError("Either wav_path or wav can be set. Please choose one.")
        
        if type(wav) is not torch.Tensor:
            raise ValueError("wav must be torch.tensor")
        if len(wav.shape) > 1:
            raise ValueError("wav must be of an 1d tensor of shape [num_samples]")

        # set up model input
        model_input = wav.unsqueeze(0)
        model_lengths = model_input.new_tensor([model_input.size(1)]).long()
        inputs = {
            self.config["model_input"]: model_input,
            self.config["model_input"] + "_lengths": model_lengths,
        }

        with torch.no_grad():
            # model forward
            if self.config["inference_mode"] == "mean_listener":
                outputs = self.model.mean_listener_inference(inputs)
            elif self.config["inference_mode"] == "mean_net":
                outputs = self.model.mean_net_inference(inputs)

        pred_mean_scores = outputs["scores"].cpu().detach().numpy()[0]
        return pred_mean_scores

def default(progress: bool = True):
    """
    The default model is the SSL-MOS model with MDF trained with all seven training sets in MOS-Bench.

    Args:
        progress - Whether to show model checkpoint load progress
    """

    # get config
    config_dst = os.path.join(torch.hub.get_dir(), "configs", os.path.basename(URLS["default"]["conf"]))
    os.makedirs(os.path.join(torch.hub.get_dir(), "configs"), exist_ok=True)
    torch.hub.download_url_to_file(URLS["default"]["conf"], dst=config_dst)
    with open(config_dst) as f:
        config = yaml.load(f, Loader=yaml.Loader)

    # init model
    if config["model_type"] == "SSLMOS":
        from sheet.models.sslmos import SSLMOS
        model = SSLMOS(
            config["model_input"],
            **config["model_params"],
        )

    # load model
    state_dict = torch.hub.load_state_dict_from_url(url=URLS["default"]["model"], map_location="cpu", progress=progress)
    model.load_state_dict(state_dict)
    model.eval()

    # send model to a Predictor wrapper
    predictor = Predictor(model, config)

    return predictor

def all8_sslmos_wavlm_large(progress: bool = True):
    """
    SSL-MOS model trained with all EIGHT training sets in MOS-Bench, as of Sep 2025.

    Args:
        progress - Whether to show model checkpoint load progress
    """

    # get config
    config_dst = os.path.join(torch.hub.get_dir(), "configs", os.path.basename(URLS["default"]["conf"]))
    os.makedirs(os.path.join(torch.hub.get_dir(), "configs"), exist_ok=True)
    torch.hub.download_url_to_file(URLS["default"]["conf"], dst=config_dst)
    with open(config_dst) as f:
        config = yaml.load(f, Loader=yaml.Loader)

    # init model
    if config["model_type"] == "SSLMOS":
        from sheet_sqa.models.sslmos import SSLMOS
        model = SSLMOS(
            config["model_input"],
            **config["model_params"],
        )

    # load model
    state_dict = torch.hub.load_state_dict_from_url(url=URLS["default"]["model"], map_location="cpu", progress=progress)
    model.load_state_dict(state_dict)
    model.eval()

    # send model to a Predictor wrapper
    predictor = Predictor(model, config)

    return predictor