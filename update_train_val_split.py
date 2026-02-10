#!/usr/bin/env python3
"""
Script to update train/val split in existing EgoMimic HDF5 files.

Usage:
    python update_train_val_split.py --hdf5_path <path_to_hdf5> --val_ratio <ratio>
    
Example inside docker:
    docker exec swati-egomimic /bin/bash -c "cd /workspace/externals/EgoMimic && python3 update_train_val_split.py --directory datasets/LBM_sim_egocentric/held_out \ 
    --output_dir datasets/LBM_sim_egocentric/train_split_combined"
"""

import h5py
import numpy as np
import argparse
import os
import shutil

VAL_RATIO = 0.05

def update_train_val_split(hdf5_path, val_ratio=VAL_RATIO, seed=42, output_path=None):
    """
    Update train/val split in an existing HDF5 file.
    
    Args:
        hdf5_path: Path to the HDF5 file
        val_ratio: Ratio of validation data (e.g., 0.2 for 20% validation)
        seed: Random seed for reproducibility
        output_path: Path to save the updated HDF5 file (if None, overwrites original)
    """
    if not os.path.exists(hdf5_path):
        print(f"Error: File {hdf5_path} does not exist!")
        return False
    
    # If output_path is specified, copy the file first
    if output_path is not None:
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
        
        print(f"Copying {hdf5_path} to {output_path}...")
        shutil.copy2(hdf5_path, output_path)
        target_path = output_path
    else:
        target_path = hdf5_path
    
    print(f"\n{'='*60}")
    print(f"Updating train/val split for: {target_path}")
    print(f"New validation ratio: {val_ratio}")
    print(f"{'='*60}\n")
    
    with h5py.File(target_path, "a") as file:
        # Get demo keys
        if "data" not in file:
            print("Error: No 'data' group found in HDF5 file!")
            return False
        
        demo_keys = [key for key in file["data"].keys() if "demo" in key]
        num_demos = len(demo_keys)
        
        if num_demos == 0:
            print("Error: No demos found in HDF5 file!")
            return False
        
        print(f"Found {num_demos} demos in the dataset")
        
        # Calculate split
        num_val = int(np.ceil(num_demos * val_ratio))
        num_train = num_demos - num_val
        
        print(f"Splitting into:")
        print(f"  - Training demos: {num_train} ({100*(1-val_ratio):.1f}%)")
        print(f"  - Validation demos: {num_val} ({100*val_ratio:.1f}%)")
        
        # Set random seed for reproducibility
        np.random.seed(seed)
        
        # Shuffle indices
        indices = np.arange(num_demos)
        np.random.shuffle(indices)
        
        # Split indices
        val_indices = indices[:num_val]
        train_indices = indices[num_val:]
        
        # Create masks
        train_mask = [f"demo_{i}" for i in sorted(train_indices)]
        val_mask = [f"demo_{i}" for i in sorted(val_indices)]
        
        print(f"\nTrain demos: {train_mask[:5]}{'...' if len(train_mask) > 5 else ''}")
        print(f"Val demos: {val_mask[:5]}{'...' if len(val_mask) > 5 else ''}")
        
        # Delete old masks if they exist
        if "mask" in file:
            print("\nRemoving old train/val masks...")
            del file["mask"]
        
        # Create new masks
        print("Creating new train/val masks...")
        mask_group = file.create_group("mask")
        mask_group.create_dataset("train", data=np.array(train_mask, dtype="S"))
        mask_group.create_dataset("valid", data=np.array(val_mask, dtype="S"))
        
        print("\n✓ Successfully updated train/val split!")
    
    return True

def batch_update_splits(directory, pattern="*.hdf5", val_ratio=0.2, seed=42, output_dir=None):
    """
    Update train/val split for all HDF5 files in a directory.
    
    Args:
        directory: Directory containing HDF5 files
        pattern: File pattern to match (default: *.hdf5)
        val_ratio: Validation ratio
        seed: Random seed
        output_dir: Output directory to save updated files (if None, overwrites originals)
    """
    import glob
    
    hdf5_files = glob.glob(os.path.join(directory, pattern))
    
    if not hdf5_files:
        print(f"No HDF5 files found in {directory} matching pattern {pattern}")
        return
    
    # Create output directory if specified
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output directory: {output_dir}")
    
    print(f"\nFound {len(hdf5_files)} HDF5 files to process")
    print(f"Files: {[os.path.basename(f) for f in hdf5_files]}\n")
    
    successful = 0
    failed = 0
    
    for hdf5_path in hdf5_files:
        print(f"\nProcessing: {os.path.basename(hdf5_path)}")
        
        # Determine output path
        if output_dir is not None:
            output_path = os.path.join(output_dir, os.path.basename(hdf5_path))
        else:
            output_path = None
        
        if update_train_val_split(hdf5_path, val_ratio, seed, output_path):
            successful += 1
        else:
            failed += 1
    
    print(f"\n{'='*60}")
    print(f"Batch Update Summary:")
    print(f"  - Successful: {successful}")
    print(f"  - Failed: {failed}")
    print(f"{'='*60}\n")

def inspect_current_split(hdf5_path):
    """
    Inspect the current train/val split of an HDF5 file.
    
    Args:
        hdf5_path: Path to the HDF5 file
    """
    if not os.path.exists(hdf5_path):
        print(f"Error: File {hdf5_path} does not exist!")
        return
    
    print(f"\n{'='*60}")
    print(f"Inspecting: {hdf5_path}")
    print(f"{'='*60}\n")
    
    with h5py.File(hdf5_path, "r") as file:
        # Get demo keys
        if "data" not in file:
            print("Error: No 'data' group found in HDF5 file!")
            return
        
        demo_keys = [key for key in file["data"].keys() if "demo" in key]
        num_demos = len(demo_keys)
        
        print(f"Total demos: {num_demos}")
        
        if "mask" in file:
            if "train" in file["mask"]:
                train_mask = [key.decode() if isinstance(key, bytes) else key 
                             for key in file["mask/train"][:]]
                print(f"Training demos: {len(train_mask)} ({100*len(train_mask)/num_demos:.1f}%)")
                print(f"  Examples: {train_mask[:5]}{'...' if len(train_mask) > 5 else ''}")
            
            if "valid" in file["mask"]:
                val_mask = [key.decode() if isinstance(key, bytes) else key 
                           for key in file["mask/valid"][:]]
                print(f"Validation demos: {len(val_mask)} ({100*len(val_mask)/num_demos:.1f}%)")
                print(f"  Examples: {val_mask[:5]}{'...' if len(val_mask) > 5 else ''}")
        else:
            print("No train/val masks found in file!")
        
        # Show sample demo structure
        if num_demos > 0:
            demo_key = demo_keys[0]
            print(f"\nSample demo structure ({demo_key}):")
            demo_group = file[f"data/{demo_key}"]
            for key in demo_group.keys():
                if isinstance(demo_group[key], h5py.Dataset):
                    print(f"  {key}: {demo_group[key].shape}")
                elif isinstance(demo_group[key], h5py.Group):
                    print(f"  {key}/ (group)")
                    for subkey in demo_group[key].keys():
                        if isinstance(demo_group[key][subkey], h5py.Dataset):
                            print(f"    {subkey}: {demo_group[key][subkey].shape}")

def main():
    parser = argparse.ArgumentParser(
        description='Update train/val split in EgoMimic HDF5 files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Update a single file with 20% validation (overwrites original)
  python update_train_val_split.py --hdf5_path datasets/LBM_sim_egocentric/processed/task.hdf5 --val_ratio 0.2
  
  # Update a single file and save to a new location
  python update_train_val_split.py --hdf5_path datasets/LBM_sim_egocentric/processed/task.hdf5 --output_path datasets/LBM_sim_egocentric/updated/task.hdf5 --val_ratio 0.2
  
  # Inspect current split without changing it
  python update_train_val_split.py --hdf5_path datasets/LBM_sim_egocentric/processed/task.hdf5 --inspect
  
  # Update all HDF5 files in a directory with 10% validation (overwrites originals)
  python update_train_val_split.py --directory datasets/LBM_sim_egocentric/processed --val_ratio 0.1
  
  # Update all HDF5 files and save to a different directory
  python update_train_val_split.py --directory datasets/LBM_sim_egocentric/processed --output_dir datasets/LBM_sim_egocentric/updated --val_ratio 0.1
  
  # Use inside docker
  docker exec swati-egomimic python3 /workspace/externals/EgoMimic/update_train_val_split.py --hdf5_path /workspace/externals/EgoMimic/datasets/LBM_sim_egocentric/processed/task.hdf5 --val_ratio 0.2
        """
    )
    
    parser.add_argument('--hdf5_path', type=str, default=None,
                        help='Path to the HDF5 file to update')
    parser.add_argument('--output_path', type=str, default=None,
                        help='Path to save the updated HDF5 file (for single file mode)')
    parser.add_argument('--directory', type=str, default=None,
                        help='Directory containing HDF5 files (for batch processing)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory to save updated files (for batch processing)')
    parser.add_argument('--pattern', type=str, default='*.hdf5',
                        help='File pattern for batch processing (default: *.hdf5)')
    parser.add_argument('--val_ratio', type=float, default=VAL_RATIO,
                        help='Validation ratio (default: 0.05 for 5%% validation)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--inspect', action='store_true',
                        help='Only inspect current split without modifying')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.hdf5_path is None and args.directory is None:
        parser.error("Must specify either --hdf5_path or --directory")
    
    if args.hdf5_path and args.directory:
        parser.error("Cannot specify both --hdf5_path and --directory")
    
    if args.output_path and args.directory:
        parser.error("Cannot use --output_path with --directory (use --output_dir instead)")
    
    if args.output_dir and args.hdf5_path:
        parser.error("Cannot use --output_dir with --hdf5_path (use --output_path instead)")
    
    if args.val_ratio < 0 or args.val_ratio > 1:
        parser.error("val_ratio must be between 0 and 1")
    
    # Process based on mode
    if args.inspect:
        if args.hdf5_path:
            inspect_current_split(args.hdf5_path)
        else:
            parser.error("--inspect requires --hdf5_path")
    elif args.hdf5_path:
        # Single file update
        success = update_train_val_split(args.hdf5_path, args.val_ratio, args.seed, args.output_path)
        if success:
            target_file = args.output_path if args.output_path else args.hdf5_path
            print("\n✓ Done! You can verify the split with:")
            print(f"  python {os.path.basename(__file__)} --hdf5_path {target_file} --inspect")
    elif args.directory:
        # Batch update
        batch_update_splits(args.directory, args.pattern, args.val_ratio, args.seed, args.output_dir)

if __name__ == "__main__":
    main()
