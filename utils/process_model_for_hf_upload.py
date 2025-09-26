#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2025 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

"""Process model to upload to HuggingFace Models.

Current checkpoints include not only model weights,
but also internal states of the optimizer.

This script saves only the model weights.
"""

import argparse
import os
import torch

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_model", required=True, type=str, help="id of the huggingface repo")
    parser.add_argument("-o", "--output_model", required=True, type=str, help="file name to download")
    return parser

def main():
    args = get_parser().parse_args()

    if os.path.islink(args.input_model):
        parts = args.input_model.split(os.sep)
        input_model = os.path.join(
            os.sep.join(parts[:-3]),
            os.readlink(args.input_model)
        )
    else:
        input_model = args.input_model
    print(f"Processing {input_model} and save {args.output_model}")

    torch.save(torch.load(input_model, weights_only=True)["model"], args.output_model)

if __name__ == "__main__":
    main()