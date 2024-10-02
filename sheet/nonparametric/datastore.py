#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2024 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

"""datastore related"""

import faiss
import h5py
import numpy as np
from scipy.special import softmax


class Datastore():
    def __init__(
        self,
        datastore_path,
        embed_dim,
        device,
    ):
        """
        Args:
            datastore_path (str): path to the datastore.
            embed_dim (int): dimension of the embed in the datastore
        """
        embeds = []
        scores = []
        paths = []
        with h5py.File(datastore_path, "r") as f:
            for hdf5_path in list(f["scores"].keys()):
                paths.append(hdf5_path)
                embeds.append(f["embeds"][hdf5_path][()])
                scores.append(f["scores"][hdf5_path][()])
        embeds = np.stack(embeds, axis=0)
        scores = np.array(scores)
        
        # build index
        index = faiss.IndexFlatL2(embed_dim)
        if device.type == 'cuda':
            # index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, index)
            index = faiss.index_cpu_to_all_gpus(index, ngpu=1)
        # else:
            # embeds = torch.tensor(embeds, device=device)
        index.add(embeds)

        self.embeds = embeds
        self.scores = scores
        self.paths = paths
        self.index = index

    def knn(self, query, k, search_only=False):
        # search
        distances, I = self.index.search(query, k)
        scores = np.stack([self.scores[row] for row in I])
        ret = {
            "distances": distances,
            "scores": scores
        }

        if search_only:
            return ret

        inv_dist = 1 / (distances + 1e-8)
        norm_dist = softmax(inv_dist, axis=1)

        mult = np.multiply(norm_dist, scores)

        final_score = np.sum(mult, axis=1)[0]

        # retrieve IDs
        ids = [[self.paths[e] for e in row] for row in I]

        ret["final_score"] = final_score
        ret["ids"] = ids

        return ret