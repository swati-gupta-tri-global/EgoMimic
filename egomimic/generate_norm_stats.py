#!/usr/bin/env python3
"""
Generate normalization stats for an existing trained model.
This computes stats from the dataset and saves them to the experiment directory.

Usage:
    python generate_norm_stats.py --exp_dir /path/to/experiment/dir
"""

import argparse
import json
import os
import pickle
import sys

# Add the repo to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from egomimic.configs import config_factory
from egomimic.utils.train_utils import expand_dataset_paths
from egomimic.utils.multi_dataset import create_multi_file_dataset
from egomimic.utils.dataset import PlaydataSequenceDataset
import robomimic.utils.obs_utils as ObsUtils


def generate_stats_from_experiment(exp_dir):
    """Generate normalization stats from an experiment directory."""

    # Load the config
    config_path = os.path.join(exp_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found at {config_path}")

    with open(config_path, "r") as f:
        ext_cfg = json.load(f)

    print(f"Loaded config from {config_path}")

    # Create config object
    config = config_factory(ext_cfg["algo_name"])
    with config.values_unlocked():
        config.update(ext_cfg)

    # Initialize observation utilities
    ObsUtils.initialize_obs_utils_with_config(config)

    # Get dataset paths from config and expand them
    dataset_paths = expand_dataset_paths(config.train.data)
    if isinstance(dataset_paths, str):
        dataset_paths = [dataset_paths]

    print(f"Dataset paths: {len(dataset_paths)} files")

    # Get obs keys from config
    obs_keys = list(set([
        obs_key
        for modality in config.observation.modalities.obs.values()
        for obs_key in modality
    ]))
    print(f"Observation keys: {obs_keys}")

    # Get dataset keys (actions)
    dataset_keys = [config.train.ac_key]
    if hasattr(config.train, 'ac_key_hand') and config.train.ac_key_hand:
        dataset_keys.append(config.train.ac_key_hand)
    print(f"Dataset keys: {dataset_keys}")

    # Create dataset directly (bypass train/val split validation)
    print("\nCreating dataset for normalization stats...")

    trainset = create_multi_file_dataset(
        hdf5_paths=dataset_paths,
        obs_keys=obs_keys,
        dataset_keys=dataset_keys,
        goal_obs_gap=config.train.goal_obs_gap,
        type=config.train.data_type,
        ac_key=config.train.ac_key,
        frame_stack=config.train.frame_stack,
        seq_length=config.train.seq_length,
        pad_frame_stack=config.train.pad_frame_stack,
        pad_seq_length=config.train.pad_seq_length,
        get_pad_mask=True,
        goal_mode=config.train.goal_mode,
        hdf5_cache_mode=config.train.hdf5_cache_mode,
        hdf5_use_swmr=config.train.hdf5_use_swmr,
        hdf5_normalize_obs=True,  # Force normalization
        filter_by_attribute=config.train.hdf5_filter_key,
        load_next_obs=config.train.hdf5_load_next_obs,
        prestacked_actions=config.train.prestacked_actions,
        hdf5_normalize_actions=config.train.hdf5_normalize_actions,
    )

    # Get and save stats for dataset 1
    stats1 = trainset.get_obs_normalization_stats()
    if stats1 is not None:
        stats_path = os.path.join(exp_dir, "ds1_norm_stats.pkl")
        with open(stats_path, "wb") as f:
            pickle.dump(stats1, f)
        print(f"\nSaved dataset 1 stats to {stats_path}")
        print(f"  Keys: {list(stats1.keys())}")
        for key, val in stats1.items():
            print(f"    {key}: mean shape {val['mean'].shape}, std shape {val['std'].shape}")
    else:
        print("WARNING: Dataset 1 returned None for normalization stats!")

    print("\nDone!")


def main():
    parser = argparse.ArgumentParser(description="Generate normalization stats for trained model")
    parser.add_argument("--exp_dir", type=str, required=True,
                        help="Path to experiment directory (contains config.json)")

    args = parser.parse_args()

    if not os.path.isdir(args.exp_dir):
        print(f"Error: {args.exp_dir} is not a directory")
        sys.exit(1)

    generate_stats_from_experiment(args.exp_dir)


if __name__ == "__main__":
    main()
