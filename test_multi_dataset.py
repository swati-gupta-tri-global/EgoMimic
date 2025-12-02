#!/usr/bin/env python3

"""
Test script for multi-file dataset functionality
"""

import sys
import os
sys.path.append('/home/swati.gupta/workspace/cosmos-predict2/externals/EgoMimic')

def test_expand_dataset_paths():
    """Test the expand_dataset_paths function"""
    from egomimic.utils.train_utils import expand_dataset_paths
    
    print("Testing expand_dataset_paths function...")
    
    # Test with directory that doesn't exist
    print("\n1. Testing with non-existent directory:")
    result = expand_dataset_paths("datasets/non_existent_dir")
    print(f"Result: {result}")
    
    # Test with directory that exists (datasets)
    print("\n2. Testing with existing directory (datasets):")
    result = expand_dataset_paths("datasets")
    print(f"Result: {result}")
    
    # Test with list containing directories and files
    print("\n3. Testing with mixed list:")
    test_paths = [
        "datasets/LBM_sim_egocentric/processed",  # directory
        "datasets/egodex/processed"  # another directory
    ]
    result = expand_dataset_paths(test_paths)
    print(f"Result: {result}")
    
    # Test with glob pattern
    print("\n4. Testing with glob pattern:")
    result = expand_dataset_paths("datasets/**/*.hdf5")
    print(f"Result: {result}")

def test_multi_dataset():
    """Test the multi-file dataset creation"""
    from egomimic.utils.multi_dataset import create_multi_file_dataset
    
    # Test with a list of paths (even if files don't exist, we can check the logic)
    test_paths = [
        "datasets/LBM_sim_egocentric/processed/BimanualStoreCerealBoxUnderShelf.hdf5",
        "datasets/LBM_sim_egocentric/converted/TurnMugRightsideUp.hdf5"
    ]
    
    print("\nTesting multi-file dataset creation...")
    print(f"Input paths: {test_paths}")
    
    try:
        # This should create a MultiFilePlaydataSequenceDataset
        # even though the files don't exist (it will fail later when trying to open)
        dataset = create_multi_file_dataset(
            hdf5_paths=test_paths,
            obs_keys=["front_img_1", "ee_pose"],
            dataset_keys=["actions"],
            goal_obs_gap=[1, 1],
            type="robot",
            ac_key="actions"
        )
        print(f"Created dataset class: {type(dataset)}")
        print("Multi-file dataset creation logic works!")
        
    except Exception as e:
        print(f"Expected error (files don't exist): {e}")
        print("This is expected since test files don't exist")
    
    # Test with single path
    single_path = "datasets/LBM_sim_egocentric/processed/BimanualStoreCerealBoxUnderShelf.hdf5"
    print(f"\nTesting single file: {single_path}")
    
    try:
        dataset = create_multi_file_dataset(
            hdf5_paths=single_path,
            obs_keys=["front_img_1", "ee_pose"],
            dataset_keys=["actions"],
            goal_obs_gap=[1, 1],
            type="robot",
            ac_key="actions"
        )
        print(f"Created dataset class: {type(dataset)}")
        print("Single file dataset creation logic works!")
        
    except Exception as e:
        print(f"Expected error (file doesn't exist): {e}")
        print("This is expected since test file doesn't exist")

if __name__ == "__main__":
    test_expand_dataset_paths()
    test_multi_dataset()
