#!/bin/bash
set -e

uv venv tools/venv
source tools/venv/bin/activate

if [[ "$1" == "train" ]]; then
    echo "Including 'train' extras..."
    uv sync --extra train --active
else
    echo "Syncing without 'train' extras..."
    uv sync --active
fi