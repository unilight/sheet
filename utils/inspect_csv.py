#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2025 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

"""
Analyze duration statistics from a CSV file.
"""

import pandas as pd
import argparse
import librosa
import multiprocessing
from functools import partial

# Function to compute duration of one audio file
def get_duration(wav_path):
    try:
        y, sr = librosa.load(wav_path, sr=None)
        return librosa.get_duration(y=y, sr=sr)
    except Exception as e:
        print(f"Error processing {wav_path}: {e}")
        return None

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Analyze duration from a CSV file.")
    parser.add_argument("csv_path", type=str, help="Path to the input CSV file")
    args = parser.parse_args()

    # Load CSV
    df = pd.read_csv(args.csv_path)

    # Drop duplicate paths to avoid redundant processing
    unique_paths = df["wav_path"].drop_duplicates().reset_index(drop=True)

    # Use multiprocessing to compute durations for unique paths
    print(f"Computing durations for {len(unique_paths)} unique files using multiprocessing...")
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        durations = pool.map(get_duration, unique_paths)

    # Create a mapping of path to duration
    path_to_duration = dict(zip(unique_paths, durations))

    # Map back to the original DataFrame
    df['duration'] = df["wav_path"].map(path_to_duration)

    # Keep only one row per unique wav_path for stats
    df_unique = df.drop_duplicates(subset=["wav_path"])

    # (1) Overall duration statistics (based on unique files)
    total_duration = df_unique['duration'].sum()
    print("=== Duration Statistics (Unique Files) ===")
    print(f"Unique Files: {len(df_unique):.0f} / {len(df):.0f} valid rows")
    print(f"Total Duration: {total_duration:.2f} seconds ({total_duration / 3600:.2f} hours)")
    print(f"Mean: {df_unique['duration'].mean():.2f}")
    print(f"Min: {df_unique['duration'].min():.2f}")
    print(f"Max: {df_unique['duration'].max():.2f}")
    print(f"Std: {df_unique['duration'].std():.2f}")


if __name__ == "__main__":
    main()