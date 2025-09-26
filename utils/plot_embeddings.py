#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2025 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

"""
2024.7.23
Compare ssl models
1. visualization
2. retrieval

"""

import argparse
import os

import faiss
import numpy as np
import scipy
import torch
from s3prl.nn import S3PRLUpstream
from tqdm import tqdm
import yaml
from sklearn.manifold import TSNE
from sheet.datasets import NonIntrusiveDataset
from sheet.utils import read_hdf5, write_hdf5
from scipy.special import softmax
import h5py

import matplotlib.pyplot as plt
import seaborn as sns

# MODEL_NAME = "all7_alignnet-1337_86000steps"

def metric(truths, preds):
    MSE = np.mean((truths - preds) ** 2)
    LCC = np.corrcoef(truths, preds)[0][1]
    SRCC = scipy.stats.spearmanr(truths.T, preds.T)[0]
    KTAU = scipy.stats.kendalltau(truths, preds)[0]
    return MSE, LCC, SRCC, KTAU


def load_ssl_model(s3prl_name):
    if s3prl_name in S3PRLUpstream.available_names():
        return S3PRLUpstream(s3prl_name)


def extract_embedding(model, waveform, waveform_lengths, idx=-1):
    all_encoder_outputs, all_encoder_outputs_lens = model(waveform, waveform_lengths)
    return torch.mean(all_encoder_outputs[idx].squeeze(0), dim=0).detach().cpu().numpy()


def run_tsne(embeds, scores, figname):
    tsne = TSNE(n_components=2, random_state=42)
    tsne = tsne.fit_transform(embeds)

    norm = plt.Normalize(1, 5)
    sm = plt.cm.ScalarMappable(cmap="Reds", norm=norm)

    plt.figure(figsize=(10, 10))
    ax = sns.scatterplot(
        x=tsne[:, 0],
        y=tsne[:, 1],
        hue=scores,
        palette=sns.color_palette("flare", as_cmap=True),
    )
    ax.get_legend().remove()
    ax.figure.colorbar(sm, ax=ax)
    plt.savefig(figname)


def run_tsne_two(embeds1, embeds2, scores1, scores2, figname):
    len1 = len(scores1)
    embeds = np.concatenate([embeds1, embeds2], axis=0)
    scores = scores1 + scores2
    tsne = TSNE(n_components=2, random_state=42)
    tsne = tsne.fit_transform(embeds)

    plt.figure(figsize=(10, 10))
    sns.scatterplot(
        x=tsne[:len1, 0],
        y=tsne[:len1, 1],
        hue=scores[:len1],
        palette=sns.color_palette("flare", as_cmap=True),
    )
    sns.scatterplot(
        x=tsne[len1:, 0],
        y=tsne[len1:, 1],
        hue=scores[len1:],
        palette=sns.color_palette("crest", as_cmap=True),
    )
    plt.savefig(figname)


def get_features(
    dataset, hdf5_filename, model, device
):
    
    embeds = []
    scores = []

    if os.path.exists(hdf5_filename):
        hdf5_file = h5py.File(hdf5_filename, "r+")
    else:
        hdf5_file = h5py.File(hdf5_filename, "w")

    # extraction
    with torch.no_grad(), tqdm(dataset, desc="[inference]") as pbar:
        for batch in pbar:
            scores.append(batch["avg_score"])

            system_id = batch["system_id"]
            sample_id = batch["sample_id"]


            hdf5_path = system_id + "_" + sample_id

            if hdf5_path in hdf5_file:
                embed = hdf5_file[hdf5_path][()]
            else:
                # set up model input and extract
                waveform = batch["waveform"].unsqueeze(0).to(device)
                waveform_lengths = waveform.new_tensor([waveform.size(1)]).long()
                embed = extract_embedding(
                    model, waveform, waveform_lengths
                )

                # save
                hdf5_file.create_dataset(hdf5_path, data=embed)
                hdf5_file.flush()
                

            # after_ft_embed_path = os.path.join(
                # model_name, dataset_name, system_id + "_" + sample_id + ".npy"
            # )
            # if os.path.isfile(after_ft_embed_path):
            #     after_ft_embed = np.load(after_ft_embed_path)
            # else:
            #     after_ft_embed = extract_embedding(
            #         after_ft_model, waveform, waveform_lengths
            #     )
            #     np.save(after_ft_embed_path, after_ft_embed)

            embeds.append(embed)
    
    hdf5_file.close()

    embeds = np.stack(embeds, axis=0)

    return embeds, scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--expdir",
        required=True,
        type=str,
        help=("experiment directory."),
    )
    parser.add_argument(
        "--train_csv",
        required=True,
        type=str,
        help=("train csv"),
    )
    args = parser.parse_args()

    # setup device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    # load config
    print("Reading config")
    config_path = os.path.join(args.expdir, "config.yml")
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.Loader)

    # get model
    print("Getting model")
    s3prl_name = config["model_params"]["s3prl_name"]
    # before_ft_model = S3PRLUpstream(s3prl_name)
    before_model = S3PRLUpstream(s3prl_name)
    after_model = S3PRLUpstream(s3prl_name)

    # load pre-trained model
    ckpt_path_part1 = os.sep.join(args.expdir.split(os.sep)[:-2])
    ckpt_path_part2 = os.readlink(os.path.join(args.expdir, "checkpoint-best.pkl"))
    ckpt_path = os.path.join(ckpt_path_part1, ckpt_path_part2)
    print("Load model from", ckpt_path)
    pt_ckpt = torch.load(ckpt_path, map_location="cpu")["model"]
    state_dict = {
        k.replace("ssl_model.", ""): v
        for k, v in pt_ckpt.items()
        if k.startswith("ssl_model")
    }
    after_model.load_state_dict(state_dict)
    before_model = before_model.eval().to(device)
    after_model = after_model.eval().to(device)

    # get features
    before_embeds = {}
    after_embeds = {}
    
    print(f"Reading dataset {args.train_csv}")
    dataset = NonIntrusiveDataset(
        csv_path=args.train_csv,
        target_sample_rate=config["sampling_rate"],
        model_input=config["model_input"],
        use_phoneme=config.get("use_phoneme", False),
        symbols=config.get("symbols", None),
        wav_only=True,
        allow_cache=False,
    )

    embed_dir = os.path.join(args.expdir, "ssl_plots")
    os.makedirs(embed_dir, exist_ok=True)
    before_embed_path = os.path.join(embed_dir, "before_embed.h5")
    after_embed_path = os.path.join(embed_dir, "after_embed.h5")
    
    print("Extracting embedding using original SSL")
    before_embeds, before_scores = get_features(dataset, before_embed_path, before_model, device)
    print("  running TSNE")
    run_tsne(before_embeds, before_scores, os.path.join(embed_dir, "before.png"))

    print("Extracting embedding using fine-tuned SSL")
    after_embeds, after_scores = get_features(dataset, after_embed_path, after_model, device)
    print("  running TSNE")
    run_tsne(after_embeds, after_scores, os.path.join(embed_dir, "after.png"))

    ##########################
    if False:
        for name1, name2 in [
            # ["all7_train", "bvcc_test"],
            # ["bvcc_train", "bvcc_test"],
            # ["bvcc_train", "somos_test"],
            # ["nisqa_train", "nisqa_livetalk"],
            # ["nisqa_train", "nisqa_for"],
            # ["bvcc_train", "nisqa_train"],
            ["all7_train", "nisqa_livetalk"],
            ["all7_train", "nisqa_for"],
            ["all7_train", "nisqa_train"],
        ]:
            print(name1, "+", name2)
            # print("TSNE: before")
            # run_tsne_two(
            #     embeds[name1][0],
            #     embeds[name2][0],
            #     embeds[name1][2],
            #     embeds[name2][2],
            #     os.path.join("before_ft", f"{name1}+{name2}.png"),
            # )
            print("TSNE: after")
            run_tsne_two(
                embeds[name1][1],
                embeds[name2][1],
                embeds[name1][2],
                embeds[name2][2],
                os.path.join(model_name, f"{name1}+{name2}.png"),
            )


if __name__ == "__main__":
    main()
