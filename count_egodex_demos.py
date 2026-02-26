#!/usr/bin/env python3
"""
Count the number of demonstrations in processed HDF5 files.

Usage:
    python count_egodex_demos.py --data_path datasets/egodex/processed
    python count_egodex_demos.py --data_path datasets/egodex/processed --verbose
"""

import argparse
import h5py
import os
from pathlib import Path
from collections import defaultdict


def count_demos_in_file(hdf5_path, use_val_split=False):
    """
    Count the number of demos in a single HDF5 file and get average length.
    
    Args:
        hdf5_path: Path to HDF5 file
        use_val_split: If True, only count demos in the validation split
        
    Returns:
        tuple: (demo_count, average_length)
            - demo_count: Number of demos found
            - average_length: Average number of timesteps per demo
    """
    try:
        with h5py.File(hdf5_path, 'r') as f:
            # Check if 'data' group exists
            if 'data' not in f:
                return 0, 0.0
            
            # Get validation mask if requested
            valid_demo_keys = None
            if use_val_split:
                if 'mask' in f and 'valid' in f['mask']:
                    # Mask contains demo names as bytes (e.g., b'demo_8')
                    # Convert to set of strings for comparison
                    valid_demo_keys = set(
                        name.decode('utf-8') if isinstance(name, bytes) else name
                        for name in f['mask']['valid'][:]
                    )
                else:
                    # If val_split requested but no mask exists, return 0
                    return 0, 0.0
            
            # Count demo_* groups in /data/ and get their lengths
            data_group = f['data']
            demo_lengths = []
            
            for key in data_group.keys():
                if key.startswith('demo_'):
                    # Skip if using val_split and this demo is not in the validation set
                    if use_val_split and valid_demo_keys is not None:
                        if key not in valid_demo_keys:
                            continue
                    
                    # Get length from actions or observations
                    demo_group = data_group[key]
                    
                    # Try to get length from actions_xyz_act first
                    if 'actions_xyz_act' in demo_group:
                        length = demo_group['actions_xyz_act'].shape[0]
                    # Fall back to checking obs/front_img_1
                    elif 'obs' in demo_group and 'front_img_1' in demo_group['obs']:
                        length = demo_group['obs']['front_img_1'].shape[0]
                    # Fall back to actions_xyz
                    elif 'actions_xyz' in demo_group:
                        length = demo_group['actions_xyz'].shape[0]
                    else:
                        # Skip if we can't determine length
                        continue
                    
                    demo_lengths.append(length)
            
            demo_count = len(demo_lengths)
            avg_length = sum(demo_lengths) / demo_count if demo_count > 0 else 0.0
            
            return demo_count, avg_length
    except Exception as e:
        print(f"Error reading {hdf5_path}: {e}")
        return 0, 0.0


def scan_directory(data_path, verbose=False, use_val_split=False):
    """
    Recursively scan directory for HDF5 files and count demos.
    
    Args:
        data_path: Root path to scan
        verbose: Print detailed information per file
        use_val_split: If True, only count demos in the validation split
        
    Returns:
        dict: Statistics organized by subfolder
    """
    data_path = Path(data_path)
    
    if not data_path.exists():
        print(f"Error: Path does not exist: {data_path}")
        return {}
    
    # Dictionary to store stats: {subfolder_name: {task_name: (demo_count, avg_length)}}
    stats = defaultdict(lambda: defaultdict(lambda: (0, 0.0)))
    
    # Find all HDF5 files recursively
    hdf5_files = list(data_path.rglob("*.hdf5"))
    
    if not hdf5_files:
        print(f"No HDF5 files found in {data_path}")
        return {}
    
    split_type = "validation split" if use_val_split else "all demos"
    print(f"Found {len(hdf5_files)} HDF5 files in {data_path}")
    print(f"Counting: {split_type}")
    print("="*80)
    
    for hdf5_file in sorted(hdf5_files):
        # Count demos in this file and get average length
        demo_count, avg_length = count_demos_in_file(hdf5_file, use_val_split=use_val_split)
        
        # Get relative path from data_path
        rel_path = hdf5_file.relative_to(data_path)
        
        # Get subfolder name (e.g., "part1", "part2")
        if len(rel_path.parts) > 1:
            subfolder = rel_path.parts[0]
        else:
            subfolder = "root"
        
        # Get task name (filename without extension)
        task_name = hdf5_file.stem
        
        # Store in stats
        stats[subfolder][task_name] = (demo_count, avg_length)
        
        if verbose:
            print(f"{rel_path}: {demo_count} demos, avg length: {avg_length:.1f} timesteps")
    
    return stats


def print_summary(stats):
    """
    Print summary statistics.
    
    Args:
        stats: Dictionary of statistics from scan_directory
            Format: {subfolder: {task: (demo_count, avg_length)}}
    """
    print("\n" + "="*80)
    print("SUMMARY BY SUBFOLDER")
    print("="*80)
    
    overall_total_demos = 0
    overall_files = 0
    all_avg_lengths = []
    
    for subfolder in sorted(stats.keys()):
        subfolder_tasks = stats[subfolder]
        
        # Calculate subfolder totals
        subfolder_total_demos = sum(count for count, _ in subfolder_tasks.values())
        subfolder_avg_lengths = [avg_len for _, avg_len in subfolder_tasks.values() if avg_len > 0]
        subfolder_avg_length = sum(subfolder_avg_lengths) / len(subfolder_avg_lengths) if subfolder_avg_lengths else 0.0
        
        num_files = len(subfolder_tasks)
        
        overall_total_demos += subfolder_total_demos
        overall_files += num_files
        all_avg_lengths.extend(subfolder_avg_lengths)
        
        print(f"\n{subfolder}/")
        print(f"  Files: {num_files}")
        print(f"  Total demos: {subfolder_total_demos}")
        print(f"  Avg demo length: {subfolder_avg_length:.1f} timesteps")
        
        # Show per-task breakdown
        print(f"  Tasks:")
        for task_name in sorted(subfolder_tasks.keys()):
            demo_count, avg_length = subfolder_tasks[task_name]
            print(f"    - {task_name}.hdf5: {demo_count} demos, avg length: {avg_length:.1f} timesteps")
    
    overall_avg_length = sum(all_avg_lengths) / len(all_avg_lengths) if all_avg_lengths else 0.0
    
    print("\n" + "="*80)
    print("OVERALL TOTALS")
    print("="*80)
    print(f"Total subfolders: {len(stats)}")
    print(f"Total HDF5 files: {overall_files}")
    print(f"Total demos: {overall_total_demos}")
    print(f"Average demos per file: {overall_total_demos / overall_files:.1f}" if overall_files > 0 else "N/A")
    print(f"Average demo length: {overall_avg_length:.1f} timesteps")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Count demonstrations in processed EgoDex HDF5 files"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to processed egodex data directory"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed information per file"
    )
    parser.add_argument(
        "--val_split",
        action="store_true",
        help="Only count demos in the validation split (requires mask/valid in HDF5)"
    )
    
    args = parser.parse_args()
    
    # Scan directory
    stats = scan_directory(args.data_path, verbose=args.verbose, use_val_split=args.val_split)
    
    if stats:
        # Print summary
        print_summary(stats)
    else:
        print("No statistics collected.")


if __name__ == "__main__":
    main()
