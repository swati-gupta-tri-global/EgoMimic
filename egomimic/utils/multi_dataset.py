import os
import h5py
import numpy as np
from copy import deepcopy
import random
from contextlib import contextmanager

import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.log_utils as LogUtils
import robomimic.utils.file_utils as FileUtils

from egomimic.utils.dataset import PlaydataSequenceDataset


class MultiFilePlaydataSequenceDataset(PlaydataSequenceDataset):
    """
    Dataset class that can handle multiple HDF5 files and concatenate them into a single dataset.
    This allows training on multiple datasets as if they were a single large dataset.
    """
    
    def __init__(
        self,
        hdf5_paths,  # List of paths instead of single path
        obs_keys,
        dataset_keys,
        goal_obs_gap,
        type,
        ac_key,
        frame_stack=1,
        seq_length=1,
        pad_frame_stack=True,
        pad_seq_length=True,
        get_pad_mask=False,
        goal_mode=None,
        hdf5_cache_mode=None,
        hdf5_use_swmr=True,
        hdf5_normalize_obs=False,
        filter_by_attribute=None,
        load_next_obs=True,
        prestacked_actions=False,
        hdf5_normalize_actions=False
    ):
        """
        Args:
            hdf5_paths (list): List of paths to hdf5 files to load
            ... (all other arguments are the same as PlaydataSequenceDataset)
        """
        
        # Store all paths
        if isinstance(hdf5_paths, str):
            hdf5_paths = [hdf5_paths]
        self.hdf5_paths = hdf5_paths
        
        # We'll use the first file to initialize the parent class
        # and then merge in the other files
        super().__init__(
            hdf5_path=hdf5_paths[0],
            obs_keys=obs_keys,
            dataset_keys=dataset_keys,
            goal_obs_gap=goal_obs_gap,
            type=type,
            ac_key=ac_key,
            frame_stack=frame_stack,
            seq_length=seq_length,
            pad_frame_stack=pad_frame_stack,
            pad_seq_length=pad_seq_length,
            get_pad_mask=get_pad_mask,
            goal_mode=goal_mode,
            hdf5_cache_mode=None,  # We'll handle caching ourselves
            hdf5_use_swmr=False,  # Disable SWMR for better compatibility
            hdf5_normalize_obs=False,  # We'll handle normalization at the end
            filter_by_attribute=filter_by_attribute,
            load_next_obs=load_next_obs,
            prestacked_actions=prestacked_actions,
            hdf5_normalize_actions=hdf5_normalize_actions
        )
        
        # Store original cache mode
        self.original_cache_mode = hdf5_cache_mode
        self.original_normalize_obs = hdf5_normalize_obs
        
        # Now we need to merge data from additional files
        if len(hdf5_paths) > 1:
            self._merge_additional_files()
            # save the merged dataset for future use
            
            
        # Skip caching for multi-file datasets to improve performance
        # Caching is disabled as it takes too long for multiple files
        print("Caching disabled for multi-file datasets to improve performance")
        self.hdf5_cache_mode = None
            
        # Apply normalization if requested
        if self.original_normalize_obs:
            self.hdf5_normalize_obs = self.original_normalize_obs
            self.normalize_obs()
    
    def _merge_additional_files(self):
        """
        Merge additional HDF5 files into the dataset.
        This modifies the demos list and other internal state to include data from all files.
        """
        print(f"Merging {len(self.hdf5_paths)} HDF5 files...")
        
        # Track which file each demo comes from
        self.demo_file_map = {}
        for demo in self.demos:
            self.demo_file_map[demo] = 0
            
        # Load additional files
        for file_idx, hdf5_path in enumerate(self.hdf5_paths[1:], 1):
            print(f"Loading file {file_idx + 1}/{len(self.hdf5_paths)}: {hdf5_path}")
            
            # Open file and get demo keys with more robust error handling
            try:
                with h5py.File(hdf5_path, 'r', swmr=False, libver='latest') as f:
                    # Get demo keys, applying filter if needed
                    if self.filter_by_attribute is not None:
                        if f"mask/{self.filter_by_attribute}" in f:
                            temp_demos = [elem.decode("utf-8") for elem in np.array(f[f"mask/{self.filter_by_attribute}"][:])]
                            print(f"  Found {len(temp_demos)} demos with filter '{self.filter_by_attribute}'")
                        else:
                            print(f"  Warning: Filter key '{self.filter_by_attribute}' not found in {hdf5_path}")
                            print(f"  Available masks: {list(f['mask'].keys()) if 'mask' in f else 'No mask group found'}")
                            # Use all demos if filter not found
                            if "data" in f:
                                temp_demos = list(f["data"].keys())
                                print(f"  Using all {len(temp_demos)} demos from data group")
                            else:
                                print(f"  Error: No 'data' group found in {hdf5_path}")
                                continue
                    else:
                        if "data" in f:
                            temp_demos = list(f["data"].keys())
                            print(f"  Found {len(temp_demos)} demos in data group")
                        else:
                            print(f"  Error: No 'data' group found in {hdf5_path}")
                            continue
                    
                    if not temp_demos:
                        print(f"  Warning: No demos found in {hdf5_path}, skipping")
                        continue
                    
                    # Sort demo keys safely
                    try:
                        inds = np.argsort([int(elem[5:]) for elem in temp_demos if elem.startswith('demo_')])
                        temp_demos = [temp_demos[i] for i in inds]
                    except (ValueError, IndexError) as e:
                        print(f"  Warning: Could not sort demos in {hdf5_path}: {e}. Using original order.")
                        # Use original order if sorting fails
                        pass
                    
                    # Rename demos to avoid conflicts (add file prefix)
                    for temp_demo in temp_demos:
                        # Create new demo name with file prefix
                        new_demo_name = f"file_{file_idx}_{temp_demo}"
                        self.demos.append(new_demo_name)
                        self.demo_file_map[new_demo_name] = file_idx
                        
                    print(f"  Successfully added {len(temp_demos)} demos with prefix 'file_{file_idx}_'")
            except (OSError, KeyError, ValueError) as e:
                print(f"Error loading file {hdf5_path}: {e}")
                continue
                    
        # Update total samples and indices
        self._update_indices()
        
        # Validate the merged dataset
        self._validate_merged_dataset()
            
        print(f"Successfully merged {len(self.hdf5_paths)} files with {len(self.demos)} total demonstrations")
    
    def _update_indices(self):
        """Update the internal indices after merging files."""
        # Clear existing indices - use parent class variable names
        self._index_to_demo_id = {}
        self._demo_id_to_start_indices = {}
        self._demo_id_to_demo_length = {}
        self.total_num_sequences = 0
        
        # Rebuild for all demos
        for demo_ind, demo_id in enumerate(self.demos):
            file_idx = self.demo_file_map[demo_id]
            
            # Get demo length from the appropriate file with error handling
            try:
                with h5py.File(self.hdf5_paths[file_idx], 'r', swmr=False, libver='latest') as f:
                    original_demo_name = demo_id.split('_', 2)[-1] if file_idx > 0 else demo_id
                    demo_length = f[f"data/{original_demo_name}"].attrs["num_samples"]
                    
                # Calculate number of sequences for this demo
                if self.pad_seq_length:
                    num_sequences = max(1, demo_length - self.seq_length + 1)
                else:
                    num_sequences = max(0, demo_length - self.seq_length + 1)
                
                # Store demo metadata using parent class structure
                self._demo_id_to_start_indices[demo_id] = self.total_num_sequences
                self._demo_id_to_demo_length[demo_id] = demo_length
                
                # Map each sequence index to its demo_id
                for seq_ind in range(num_sequences):
                    self._index_to_demo_id[self.total_num_sequences] = demo_id
                    self.total_num_sequences += 1
                    
            except (OSError, KeyError, ValueError) as e:
                print(f"Error processing demo {demo_id}: {e}")
                continue
                
        print(f"Total sequences across all files: {self.total_num_sequences}")
        print(f"Index mapping size: {len(self._index_to_demo_id)}")
        print(f"Demo count: {len(self._demo_id_to_start_indices)}")
    
    def _validate_merged_dataset(self):
        """Validate that the merged dataset has consistent keys across all demos."""
        print("Validating merged dataset...")
        
        # Sample a few demos from each file to check key consistency
        file_demo_samples = {}
        for demo_id in self.demos[:min(10, len(self.demos))]:  # Check first 10 demos
            file_idx = self.demo_file_map[demo_id]
            if file_idx not in file_demo_samples:
                file_demo_samples[file_idx] = []
            file_demo_samples[file_idx].append(demo_id)
        
        # Check what obs keys are available in each file
        all_obs_keys = set()
        file_obs_keys = {}
        file_action_shapes = {}  # Track action shapes across files
        
        for file_idx, demo_list in file_demo_samples.items():
            demo_id = demo_list[0]  # Check first demo from each file
            file_idx = self.demo_file_map[demo_id]
            original_demo_name = demo_id.split('_', 2)[-1] if file_idx > 0 else demo_id
            
            try:
                with h5py.File(self.hdf5_paths[file_idx], 'r', swmr=False, libver='latest') as f:
                    if f"data/{original_demo_name}/obs" in f:
                        obs_keys = list(f[f"data/{original_demo_name}/obs"].keys())
                        file_obs_keys[file_idx] = obs_keys
                        all_obs_keys.update(obs_keys)
                        print(f"  File {file_idx} obs keys: {obs_keys}")
                    else:
                        print(f"  Warning: No obs group found in file {file_idx}, demo {original_demo_name}")
                    
                    # Check action dimensions
                    action_keys_to_check = [self.ac_key, 'actions', 'actions_joints_act', 'actions_xyz_act']
                    for ac_key in action_keys_to_check:
                        action_path = f"data/{original_demo_name}/{ac_key}"
                        if action_path in f:
                            action_shape = f[action_path].shape
                            file_action_shapes[file_idx] = {ac_key: action_shape}
                            print(f"  File {file_idx} action '{ac_key}' shape: {action_shape}")
                            break
                    else:
                        print(f"  Warning: No action data found in file {file_idx}, demo {original_demo_name}")
                        
            except Exception as e:
                print(f"  Error checking file {file_idx}: {e}")
                continue
        
        # Check for key consistency
        if len(file_obs_keys) > 1:
            print("Checking obs key consistency across files...")
            reference_keys = set(list(file_obs_keys.values())[0])
            for file_idx, keys in file_obs_keys.items():
                key_set = set(keys)
                missing_keys = reference_keys - key_set
                extra_keys = key_set - reference_keys
                if missing_keys:
                    print(f"  File {file_idx} missing keys: {missing_keys}")
                if extra_keys:
                    print(f"  File {file_idx} has extra keys: {extra_keys}")
        
        # Check action shape consistency
        # if len(file_action_shapes) > 1:
        #     print("Checking action shape consistency across files...")
        #     reference_file = list(file_action_shapes.keys())[0]
        #     reference_shapes = file_action_shapes[reference_file]
            
        #     for file_idx, action_shapes in file_action_shapes.items():
        #         if file_idx == reference_file:
        #             continue
                    
        #         for ac_key, shape in action_shapes.items():
        #             if ac_key in reference_shapes:
        #                 ref_shape = reference_shapes[ac_key]
        #                 if shape != ref_shape:
        #                     print(f"  ACTION SHAPE MISMATCH!")
        #                     print(f"    File {reference_file} '{ac_key}': {ref_shape}")
        #                     print(f"    File {file_idx} '{ac_key}': {shape}")
        #                     # print(f"    This could cause the ac_dim=100 issue!")
        
        print(f"Dataset validation complete. Found {len(all_obs_keys)} unique observation keys across all files.")
        print(f"All obs keys: {sorted(all_obs_keys)}")
                
        print(f"Total sequences across all files: {len(self._index_to_demo_id)}")
    
    def _get_from_file(self, ep, key):
        """Get data directly from the appropriate HDF5 file."""
        file_idx = self.demo_file_map[ep]
        original_demo_name = ep.split('_', 2)[-1] if file_idx > 0 else ep
        
        # Use more robust file opening with error handling
        # Disable SWMR for better compatibility
        try:
            with h5py.File(self.hdf5_paths[file_idx], 'r', swmr=False, libver='latest') as f:
                # Try different possible key locations
                possible_keys = [
                    f"data/{original_demo_name}/{key}",
                    f"data/{original_demo_name}/obs/{key}",
                    f"data/{original_demo_name}/next_obs/{key}",
                    f"data/{original_demo_name}/actions/{key}" if key == 'actions' else None,
                ]
                
                # Filter out None values
                possible_keys = [k for k in possible_keys if k is not None]
                
                for hd5key in possible_keys:
                    if hd5key in f:
                        return f[hd5key][()]  # Load the actual data
                
                # If not found in standard locations, print available keys for debugging
                if f"data/{original_demo_name}" in f:
                    available_keys = list(f[f"data/{original_demo_name}"].keys())
                    if 'obs' in available_keys and f"data/{original_demo_name}/obs" in f:
                        obs_keys = list(f[f"data/{original_demo_name}/obs"].keys())
                        print(f"Available obs keys in {original_demo_name}: {obs_keys}")
                    print(f"Requested key '{key}' not found in {original_demo_name}. Available top-level keys: {available_keys}")
                else:
                    print(f"Demo {original_demo_name} not found in {self.hdf5_paths[file_idx]}")
                    
                return None
        except (OSError, KeyError, ValueError) as e:
            print(f"Error reading {key} from {ep} in file {self.hdf5_paths[file_idx]}: {e}")
            return None
    
    def get_dataset_for_ep(self, ep, key):
        """Override to handle multiple files. Always read from file for better performance."""
        # Always read directly from file to avoid caching overhead
        return self._get_from_file(ep, key)
    
    def __len__(self):
        """Return total number of sequences across all files."""
        return self.total_num_sequences
    
    def close_and_delete_hdf5_handle(self):
        """Override to handle multiple files.""" 
        # No cache to clear since caching is disabled for performance
        # Parent class handles the main file
        super().close_and_delete_hdf5_handle()


def create_multi_file_dataset(
    hdf5_paths,
    obs_keys,
    dataset_keys, 
    goal_obs_gap,
    type,
    ac_key,
    frame_stack=1,
    seq_length=1,
    pad_frame_stack=True,
    pad_seq_length=True,
    get_pad_mask=False,
    goal_mode=None,
    hdf5_cache_mode=None,
    hdf5_use_swmr=True,
    hdf5_normalize_obs=False,
    filter_by_attribute=None,
    load_next_obs=True,
    prestacked_actions=False,
    hdf5_normalize_actions=False
):
    """
    Factory function to create a multi-file dataset.
    
    Args:
        hdf5_paths (list or str): List of HDF5 file paths, or single path
        ... (other args same as PlaydataSequenceDataset)
    
    Returns:
        Dataset instance (either MultiFilePlaydataSequenceDataset or PlaydataSequenceDataset)
    """
    if isinstance(hdf5_paths, str):
        hdf5_paths = [hdf5_paths]
    
    if len(hdf5_paths) == 1:
        # Use regular single-file dataset
        return PlaydataSequenceDataset(
            hdf5_path=hdf5_paths[0],
            obs_keys=obs_keys,
            dataset_keys=dataset_keys,
            goal_obs_gap=goal_obs_gap,
            type=type,
            ac_key=ac_key,
            frame_stack=frame_stack,
            seq_length=seq_length,
            pad_frame_stack=pad_frame_stack,
            pad_seq_length=pad_seq_length,
            get_pad_mask=get_pad_mask,
            goal_mode=goal_mode,
            hdf5_cache_mode=hdf5_cache_mode,
            hdf5_use_swmr=hdf5_use_swmr,
            hdf5_normalize_obs=hdf5_normalize_obs,
            filter_by_attribute=filter_by_attribute,
            load_next_obs=load_next_obs,
            prestacked_actions=prestacked_actions,
            hdf5_normalize_actions=hdf5_normalize_actions
        )
    else:
        # Use multi-file dataset
        return MultiFilePlaydataSequenceDataset(
            hdf5_paths=hdf5_paths,
            obs_keys=obs_keys,
            dataset_keys=dataset_keys,
            goal_obs_gap=goal_obs_gap,
            type=type,
            ac_key=ac_key,
            frame_stack=frame_stack,
            seq_length=seq_length,
            pad_frame_stack=pad_frame_stack,
            pad_seq_length=pad_seq_length,
            get_pad_mask=get_pad_mask,
            goal_mode=goal_mode,
            hdf5_cache_mode=hdf5_cache_mode,
            hdf5_use_swmr=hdf5_use_swmr,
            hdf5_normalize_obs=hdf5_normalize_obs,
            filter_by_attribute=filter_by_attribute,
            load_next_obs=load_next_obs,
            prestacked_actions=prestacked_actions,
            hdf5_normalize_actions=hdf5_normalize_actions
        )
