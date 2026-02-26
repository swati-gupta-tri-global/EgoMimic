#!/usr/bin/env python3
"""
Merge multiple HDF5 files in robomimic/EgoMimic format into a single combined file.

Usage:
    python merge_hdf5_files.py --input file1.hdf5 file2.hdf5 file3.hdf5 --output combined.hdf5
    python merge_hdf5_files.py --input-dir /path/to/hdf5/files/ --output combined.hdf5
    python merge_hdf5_files.py --input-dir /path/to/hdf5/files/ --output combined.hdf5 --pattern "*_robot.hdf5"

python merge_hdf5_files.py --input-dir datasets/egomimic/hand/ --output datasets/egomimic/hand_combined.hdf5

Found 50 demos
    Copying demo_0 -> demo_0 [TRAIN]
    Copying demo_1 -> demo_1 [TRAIN]
    Copying demo_10 -> demo_2 [TRAIN]
    Copying demo_100 -> demo_3 [TRAIN]
    Copying demo_101 -> demo_4 [VALID]
    Copying demo_102 -> demo_5 [VALID]
"""

import h5py
import numpy as np
import argparse
import os
from pathlib import Path
from typing import List, Dict
import glob


def get_hdf5_files(input_paths: List[str] = None, input_dir: str = None, pattern: str = "*.hdf5") -> List[str]:
    """
    Get list of HDF5 files from either explicit paths or directory scan.

    Args:
        input_paths: List of explicit file paths
        input_dir: Directory to scan for HDF5 files
        pattern: Glob pattern for files in directory (default: *.hdf5)

    Returns:
        List of HDF5 file paths
    """
    if input_paths:
        # Validate explicit paths
        for path in input_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Input file not found: {path}")
            if not path.endswith('.hdf5'):
                raise ValueError(f"Not an HDF5 file: {path}")
        return sorted(input_paths)

    elif input_dir:
        # Scan directory
        if not os.path.isdir(input_dir):
            raise NotADirectoryError(f"Directory not found: {input_dir}")

        search_path = os.path.join(input_dir, pattern)
        files = sorted(glob.glob(search_path))

        if not files:
            raise FileNotFoundError(f"No HDF5 files found matching pattern '{pattern}' in {input_dir}")

        print(f"Found {len(files)} HDF5 files in {input_dir}:")
        for i, f in enumerate(files, 1):
            print(f"  {i}. {os.path.basename(f)}")

        return files

    else:
        raise ValueError("Must provide either --input or --input-dir")


def copy_demo_data(src_file: h5py.File, dst_file: h5py.File, src_demo_name: str, dst_demo_name: str):
    """
    Copy a single demo from source to destination file with new name.

    Args:
        src_file: Source HDF5 file handle
        dst_file: Destination HDF5 file handle
        src_demo_name: Source demo name (e.g., 'demo_0')
        dst_demo_name: Destination demo name (e.g., 'demo_5')
    """
    src_path = f"data/{src_demo_name}"

    if src_path not in src_file:
        raise KeyError(f"Demo {src_demo_name} not found in source file")

    # Copy entire demo group recursively
    src_file.copy(src_path, dst_file['data'], name=dst_demo_name)


def merge_hdf5_files(input_files: List[str], output_file: str, verbose: bool = True):
    """
    Merge multiple HDF5 files into a single combined file.

    Args:
        input_files: List of input HDF5 file paths
        output_file: Output HDF5 file path
        verbose: Print progress information
    """
    if os.path.exists(output_file):
        response = input(f"Output file {output_file} already exists. Overwrite? [y/N]: ")
        if response.lower() != 'y':
            print("Aborted.")
            return
        os.remove(output_file)

    # Statistics
    total_demos = 0
    train_demos = []
    valid_demos = []
    demo_counter = 0

    # Create output file
    with h5py.File(output_file, 'w') as out_f:
        # Create data and mask groups
        out_f.create_group('data')
        out_f.create_group('mask')

        # Process each input file
        for file_idx, input_file in enumerate(input_files):
            if verbose:
                print(f"\nProcessing file {file_idx + 1}/{len(input_files)}: {os.path.basename(input_file)}")

            with h5py.File(input_file, 'r') as in_f:
                # Get all demos in this file
                if 'data' not in in_f:
                    print(f"  WARNING: No 'data' group found in {input_file}, skipping")
                    continue

                demos = sorted(in_f['data'].keys())
                if verbose:
                    print(f"  Found {len(demos)} demos")

                # Get train/valid masks if they exist
                file_train_demos = set()
                file_valid_demos = set()

                if 'mask' in in_f:
                    if 'train' in in_f['mask']:
                        # Handle both bytes and string types
                        train_data = in_f['mask/train'][()]
                        file_train_demos = set(
                            x.decode('utf-8') if isinstance(x, bytes) else str(x)
                            for x in train_data
                        )
                    if 'valid' in in_f['mask']:
                        # Handle both bytes and string types
                        valid_data = in_f['mask/valid'][()]
                        file_valid_demos = set(
                            x.decode('utf-8') if isinstance(x, bytes) else str(x)
                            for x in valid_data
                        )

                # Copy each demo
                for src_demo_name in demos:
                    dst_demo_name = f"demo_{demo_counter}"

                    if verbose:
                        print(f"    Copying {src_demo_name} -> {dst_demo_name}", end='')

                    # Copy demo data
                    copy_demo_data(in_f, out_f, src_demo_name, dst_demo_name)

                    # Track train/valid split
                    if src_demo_name in file_train_demos:
                        train_demos.append(dst_demo_name)
                        if verbose:
                            print(" [TRAIN]")
                    elif src_demo_name in file_valid_demos:
                        valid_demos.append(dst_demo_name)
                        if verbose:
                            print(" [VALID]")
                    else:
                        # Default to train if no mask specified
                        train_demos.append(dst_demo_name)
                        if verbose:
                            print(" [TRAIN (default)]")

                    demo_counter += 1
                    total_demos += 1

        # Write train/valid masks
        if train_demos:
            out_f['mask'].create_dataset('train', data=np.array(train_demos, dtype='S'))
            if verbose:
                print(f"\nTrain demos: {len(train_demos)}")

        if valid_demos:
            out_f['mask'].create_dataset('valid', data=np.array(valid_demos, dtype='S'))
            if verbose:
                print(f"Valid demos: {len(valid_demos)}")

        # Copy env_args from first file if it exists
        with h5py.File(input_files[0], 'r') as first_file:
            if 'env_args' in first_file.attrs:
                out_f.attrs['env_args'] = first_file.attrs['env_args']

    if verbose:
        print(f"\n{'='*60}")
        print(f"Successfully merged {len(input_files)} files into {output_file}")
        print(f"Total demos: {total_demos}")
        print(f"  Train: {len(train_demos)}")
        print(f"  Valid: {len(valid_demos)}")
        print(f"{'='*60}")


def verify_merged_file(output_file: str):
    """
    Verify the merged HDF5 file structure.

    Args:
        output_file: Path to merged HDF5 file
    """
    print(f"\nVerifying merged file: {output_file}")

    with h5py.File(output_file, 'r') as f:
        # Check top-level structure
        assert 'data' in f, "Missing 'data' group"
        assert 'mask' in f, "Missing 'mask' group"

        # Count demos
        demos = sorted(f['data'].keys())
        print(f"  Total demos: {len(demos)}")

        # Check masks
        if 'train' in f['mask']:
            train_data = f['mask/train'][()]
            train_demos = [x.decode('utf-8') if isinstance(x, bytes) else str(x) for x in train_data]
            print(f"  Train mask: {len(train_demos)} demos")
            if len(train_demos) <= 10:
                print(f"    {train_demos}")

        if 'valid' in f['mask']:
            valid_data = f['mask/valid'][()]
            valid_demos = [x.decode('utf-8') if isinstance(x, bytes) else str(x) for x in valid_data]
            print(f"  Valid mask: {len(valid_demos)} demos")
            if len(valid_demos) <= 10:
                print(f"    {valid_demos}")

        # Check sequential naming
        expected_names = [f"demo_{i}" for i in range(len(demos))]
        if demos == expected_names:
            print(f"  ✓ Demo naming is sequential (demo_0 to demo_{len(demos)-1})")
        else:
            print(f"  ✗ WARNING: Demo naming is not sequential!")

        # Sample first demo structure
        if demos:
            demo = demos[0]
            print(f"\n  Sample demo structure ({demo}):")

            def print_structure(group, indent=4):
                for key in group.keys():
                    item = group[key]
                    if isinstance(item, h5py.Group):
                        print(f"{' '*indent}{key}/")
                        print_structure(item, indent + 2)
                    elif isinstance(item, h5py.Dataset):
                        print(f"{' '*indent}{key}: {item.shape} {item.dtype}")

            print_structure(f[f'data/{demo}'])

    print(f"\n✓ Verification complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Merge multiple HDF5 files in robomimic/EgoMimic format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Merge specific files
  python merge_hdf5_files.py --input file1.hdf5 file2.hdf5 file3.hdf5 --output combined.hdf5

  # Merge all HDF5 files in a directory
  python merge_hdf5_files.py --input-dir /path/to/files/ --output combined.hdf5

  # Merge files matching a pattern
  python merge_hdf5_files.py --input-dir /path/to/files/ --pattern "*_robot.hdf5" --output combined.hdf5

  # Verify without merging
  python merge_hdf5_files.py --verify combined.hdf5
        """
    )

    parser.add_argument('--input', nargs='+', help='Input HDF5 files to merge')
    parser.add_argument('--input-dir', help='Directory containing HDF5 files to merge')
    parser.add_argument('--pattern', default='*.hdf5', help='Glob pattern for files in input-dir (default: *.hdf5)')
    parser.add_argument('--output', help='Output merged HDF5 file path')
    parser.add_argument('--verify', help='Verify an existing merged HDF5 file')
    parser.add_argument('--quiet', action='store_true', help='Suppress progress output')

    args = parser.parse_args()

    # Verify mode
    if args.verify:
        verify_merged_file(args.verify)
        return

    # Merge mode
    if not args.output:
        parser.error("--output is required (unless using --verify)")

    if args.input:
        print ("Input files to merge: ", args.input)
    # Get input files
    input_files = get_hdf5_files(
        input_paths=args.input,
        input_dir=args.input_dir,
        pattern=args.pattern
    )

    if len(input_files) < 2:
        print("WARNING: Only one input file found. Merge requires at least 2 files.")
        response = input("Continue anyway? [y/N]: ")
        if response.lower() != 'y':
            print("Aborted.")
            return

    # Merge files
    merge_hdf5_files(input_files, args.output, verbose=not args.quiet)

    # Auto-verify
    if not args.quiet:
        verify_merged_file(args.output)


if __name__ == "__main__":
    main()
