#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2023 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

import argparse
from huggingface_hub import hf_hub_download

def get_parser():
    parser = argparse.ArgumentParser(description="download files from huggingface hub.")
    parser.add_argument("--repo_id", required=True, type=str, help="id of the huggingface repo")
    parser.add_argument("--filename", required=True, type=str, help="file name to download")
    parser.add_argument("--outdir", required=True, type=str, help="directory to save the downloaded file")
    return parser

def main():
    args = get_parser().parse_args()

    hf_hub_download(repo_id=args.repo_id,
                    filename=args.filename,
                    local_dir=args.outdir)

if __name__ == "__main__":
    main()