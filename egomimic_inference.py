#!/usr/bin/env python3
r"""
EgoMimic Inference Script with RolloutPolicy

This script loads a trained EgoMimic model and performs rollouts to get action predictions
from visual observations. It creates the RolloutPolicy wrapper and demonstrates how to
extract joint actions, XYZ actions, and gripper commands.

Usage:
docker exec -it swati-egomimic python /workspace/externals/EgoMimic/egomimic_inference.py --ckpt_path /workspace/externals/EgoMimic/trained_models_highlevel/test/None_DT_2025-11-22-18-26-12/models/model_epoch_epoch=139.ckpt \ 
    --dataset_path /workspace/externals/EgoMimic/datasets/egodex/processed/part1/clean_cups.hdf5 \ 
    --output_dir /workspace/externals/EgoMimic/inference_output_egodx --num_frames 200 --data_type 1

python egomimic_inference.py --ckpt_path /workspace/externals/EgoMimic/trained_models_highlevel/test/None_DT_2026-01-12-18-04-24/models/model_epoch_epoch=49.ckpt  --dataset_path datasets/LBM_sim_egocentric/train_split/PutBananaOnSaucer.hdf5 --output_dir inference_output_lbm --num_frames 200 --data_type 0 --visualize

docker exec -it swati-egomimic python /workspace/externals/EgoMimic/egomimic_inference.py --ckpt_path /workspace/externals/EgoMimic/trained_models_highlevel/test/None_DT_2026-01-06-22-03-51/models/model_epoch_epoch=19.ckpt     --dataset_path /workspace/externals/EgoMimic/datasets/LBM_sim_egocentric/processed/BimanualHangMugsOnMugHolderFromDryingRack.hdf5     --output_dir /workspace/externals/EgoMimic/inference_output --num_frames 200 --data_type 0
docker exec -it swati-egomimic python /workspace/externals/EgoMimic/egomimic_inference.py --ckpt_path /workspace/externals/EgoMimic/trained_models_highlevel/test/None_DT_2026-01-06-22-03-51/models/model_epoch_epoch=19.ckpt \ 
    --dataset_path /workspace/externals/EgoMimic/datasets/egodex/processed/part1/clean_cups.hdf5 \ 
    --output_dir /workspace/externals/EgoMimic/inference_output_egodex --num_frames 200 --data_type 1

# Model-2 (LBM ID + AVP) held-out eval
docker exec -it swati-egomimic python /workspace/externals/EgoMimic/egomimic_inference.py --ckpt_path /workspace/externals/EgoMimic/trained_models_highlevel/test/None_DT_2026-01-15-19-13-11/models/model_epoch_epoch=169.ckpt \ 
    --dataset_path /workspace/externals/EgoMimic/datasets/LBM_sim_egocentric/held_out \ 
    --output_dir /workspace/externals/EgoMimic/inference_output_lbm_val --num_frames 500 --num_demos=-1 --val_split --data_type 0   

# Model-1 (LBM ID) held-out eval
docker exec -it swati-egomimic python /workspace/externals/EgoMimic/egomimic_inference.py --ckpt_path /workspace/externals/EgoMimic/trained_models_highlevel/test/None_DT_2026-01-15-01-59-15/models/model_epoch_epoch=169.ckpt \ 
    --dataset_path /workspace/externals/EgoMimic/datasets/LBM_sim_egocentric/held_out \ 
    --output_dir /workspace/externals/EgoMimic/inference_output_lbm_val --num_frames 500 --num_demos=-1 --val_split --data_type 0   

"""

import argparse
import json
import os
import pickle
import numpy as np
import torch
import h5py
import cv2
# from pathlib import Path

import robomimic.utils.obs_utils as ObsUtils

# Import EgoMimic modules
from egomimic.utils.val_utils import draw_both_actions_on_frame, write_video_safe
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from egomimic.utils.val_utils import EXTRINSICS_LEFT, ARIA_INTRINSICS

class EgoMimicRolloutPolicy:
    """
    Custom policy wrapper for EgoMimic that implements get_action using forward_eval
    """
    def __init__(self, policy, obs_normalization_stats=None, data_type=0,
                 hdf5_normalize_obs=False, hdf5_normalize_actions=False,
                 imagenet_normalize_images=False):
        self.policy = policy
        self.obs_normalization_stats = obs_normalization_stats
        self.data_type = data_type  # 0 = robot, 1 = hand
        self._step_counter = 0
        self.hdf5_normalize_obs = hdf5_normalize_obs
        self.hdf5_normalize_actions = hdf5_normalize_actions
        self.imagenet_normalize_images = imagenet_normalize_images

    def __call__(self, obs_dict):
        return self.get_action(obs_dict)

    def _normalize_obs(self, obs, key):
        """Normalize a single observation using loaded stats."""
        if self.obs_normalization_stats is None or key not in self.obs_normalization_stats:
            return obs

        mean = self.obs_normalization_stats[key]["mean"][0]
        std = self.obs_normalization_stats[key]["std"][0]

        # Handle shape broadcasting
        m_num_dims = len(mean.shape)
        shape_len_diff = len(obs.shape) - m_num_dims
        reshape_padding = tuple([1] * shape_len_diff)
        mean = mean.reshape(reshape_padding + tuple(mean.shape))
        std = std.reshape(reshape_padding + tuple(std.shape))

        if isinstance(obs, torch.Tensor) and isinstance(mean, np.ndarray):
            mean = torch.from_numpy(mean).to(obs.device).float()
            std = torch.from_numpy(std).to(obs.device).float()

        return (obs - mean) / std

    def get_action(self, obs_dict, goal_dict=None):
        """
        Get action prediction using forward_eval method
        """
        import torch
        from collections import OrderedDict
        # Prepare batch format for forward_eval
        batch = OrderedDict()
        # Create obs sub-dictionary
        batch["obs"] = OrderedDict()
        # Add observations to batch["obs"]
        for key, value in obs_dict.items():
            if isinstance(value, np.ndarray):
                batch["obs"][key] = torch.from_numpy(value).float().unsqueeze(0)
            else:
                batch["obs"][key] = value.unsqueeze(0) if value.dim() == 1 else value
        # Add required fields
        batch["type"] = torch.tensor([self.data_type], dtype=torch.long)  # 0 = robot, 1 = hand
        # pad_mask needs to be (batch_size, sequence_length) - using 100 as sequence length
        batch["obs"]["pad_mask"] = torch.ones((1, 100), dtype=torch.float32)  # Not padded
        # Move to device if available
        device = next(self.policy.nets['policy'].parameters()).device
        batch["type"] = batch["type"].to(device)
        for key in batch["obs"]:
            if hasattr(batch["obs"][key], 'to'):
                batch["obs"][key] = batch["obs"][key].to(device)

        # Normalize observations if enabled (matching training behavior)
        if self.hdf5_normalize_obs and self.obs_normalization_stats is not None:
            for key in batch["obs"]:
                if key == "pad_mask":
                    continue
                # Skip RGB keys - they are handled separately via process_obs_dict
                if key in self.obs_normalization_stats:
                    batch["obs"][key] = self._normalize_obs(batch["obs"][key], key)

        # Get unnorm_stats for action unnormalization (only if actions were normalized during training)
        # DEBUG: Set to None to skip action unnormalization and see raw model output
        # unnorm_stats = None  # UNCOMMENT THIS LINE TO SKIP ACTION UNNORMALIZATION
        unnorm_stats = self.obs_normalization_stats if self.hdf5_normalize_actions else None

        # DEBUG: Print normalization info (first frame only)
        if self._step_counter == 0:
            print(f"\n=== NORMALIZATION DEBUG ===")
            print(f"hdf5_normalize_obs: {self.hdf5_normalize_obs}")
            print(f"hdf5_normalize_actions: {self.hdf5_normalize_actions}")
            print(f"unnorm_stats keys: {list(unnorm_stats.keys()) if unnorm_stats else None}")
            # Print actual mean/std for actions_xyz_act
            if unnorm_stats and 'actions_xyz_act' in unnorm_stats:
                xyz_stats = unnorm_stats['actions_xyz_act']
                print(f"actions_xyz_act mean shape: {xyz_stats['mean'].shape}")
                print(f"actions_xyz_act std shape: {xyz_stats['std'].shape}")
                # Print first few values of mean/std (first timestep, 6 values for left+right xyz)
                print(f"actions_xyz_act mean[0,:6]: {xyz_stats['mean'][0, 0, :6]}")
                print(f"actions_xyz_act std[0,:6]: {xyz_stats['std'][0, 0, :6]}")
        # Run forward_eval
        with torch.no_grad():
            predictions = self.policy.forward_eval(batch, unnorm_stats)
            
        # Debug: Print available prediction keys
        # Available prediction keys: ['actions_joints_act', 'actions_xyz_act']
        # print(f"Available prediction keys: {list(predictions.keys())}")
        # print(f"Expected action key: {self.policy.ac_key}")
            
        # Increment step counter
        self._step_counter += 1

        # For robot data, return both joint and XYZ actions if both are available
        if 'actions_joints_act' in predictions and 'actions_xyz_act' in predictions:
            # print("Robot data: Returning both joint and XYZ actions")
            return {
                'actions_joints_act': predictions['actions_joints_act'][0].cpu().numpy(),  # Remove batch dimension
                'actions_xyz_act': predictions['actions_xyz_act'][0].cpu().numpy()
            }
        
        # Otherwise, extract single action using ac_key
        ac_key = self.policy.ac_key
        if ac_key not in predictions:
            # For hand data, try alternative action keys
            possible_keys = [k for k in predictions.keys() if 'action' in k.lower()]
            print(f"Action key '{ac_key}' not found. Possible action keys: {possible_keys}")
            if possible_keys:
                ac_key = possible_keys[0]
                print(f"Using action key: {ac_key}")
            else:
                raise KeyError(f"No action key found in predictions: {list(predictions.keys())}")
                
        action = predictions[ac_key][0]  # Remove batch dimension
        
        return action.cpu().numpy()
    
    def reset(self):
        self._step_counter = 0
        if hasattr(self.policy, 'reset'):
            self.policy.reset()


def parse_egomimic_actions(action):
    """
    Parse EgoMimic action output into components.
    
    Args:
        action: Raw action output from model (numpy array)
        
    Returns:
        dict with parsed actions
        
    Action format for LBM data (based on process_LBM_sim_egocentric_to_egomimic.py):
    - action_dim=16: [left_joint(7), left_gripper(1), right_joint(7), right_gripper(1)]
    - action_dim=6: [left_xyz(3), right_xyz(3)]
    """
    action = np.array(action)
    print(f"[DEBUG] Raw action shape: {action.shape}")
    print(f"[DEBUG] Raw action: {action}")
    
    # Handle sequence predictions (T, action_dim) vs single predictions (action_dim,)
    if len(action.shape) == 2:
        # Sequence prediction - take the first timestep or last timestep
        action = action[0]  # or action[-1] for last timestep
        print(f"[DEBUG] Using first timestep from sequence: {action.shape}")
    
    action_dim = len(action)
    print(f"[DEBUG] Action dimension: {action_dim}")
    
    # Parse based on expected EgoMimic action structure for LBM data
    if action_dim == 16:
        # Joint + Gripper actions (7+1 per arm) - LBM bimanual format
        return {
            "joint_actions": {
                "left": action[0:7].tolist(),
                "right": action[8:15].tolist(),
                "combined": action[[0,1,2,3,4,5,6,8,9,10,11,12,13,14]].tolist()
            },
            "gripper_actions": {
                "left": action[7:8].tolist(),
                "right": action[15:16].tolist(),
                "combined": action[[7,15]].tolist()
            },
            "xyz_actions": None
        }
    elif action_dim == 6:
        # Only XYZ actions (3 per arm)
        return {
            "joint_actions": None, 
            "xyz_actions": {
                "left": action[0:3].tolist(),
                "right": action[3:6].tolist(), 
                "combined": action[0:6].tolist()
            },
            "gripper_actions": None
        }
    else:
        print(f"[WARNING] Unexpected action dimension: {action_dim}")
        return {
            "joint_actions": None,
            "xyz_actions": None, 
            "gripper_actions": None,
            "raw_action": action.tolist() if hasattr(action, 'tolist') else action
        }

def load_normalization_stats(ckpt_path, data_type=0):
    """
    Load normalization stats from the checkpoint directory.

    Args:
        ckpt_path: Path to model checkpoint
        data_type: 0 for robot, 1 for hand

    Returns:
        normalization_stats: Dictionary with normalization statistics, or None if not found
    """
    # Checkpoint is at: .../models/model_epoch_epoch=X.ckpt
    # Stats are at: .../ds1_norm_stats.pkl (robot) or .../ds2_norm_stats.pkl (hand)
    ckpt_dir = os.path.dirname(ckpt_path)  # .../models/
    exp_dir = os.path.dirname(ckpt_dir)    # .../

    # Select appropriate stats file based on data_type
    if data_type == 0:  # robot
        stats_file = os.path.join(exp_dir, "ds1_norm_stats.pkl")
    else:  # hand
        stats_file = os.path.join(exp_dir, "ds2_norm_stats.pkl")

    if os.path.exists(stats_file):
        print(f"Loading normalization stats from: {stats_file}")
        with open(stats_file, "rb") as f:
            normalization_stats = pickle.load(f)

        # Check if loaded stats are valid (not None and is a dict)
        if normalization_stats is not None and isinstance(normalization_stats, dict):
            print(f"Loaded normalization stats for keys: {list(normalization_stats.keys())}")
            return normalization_stats
        else:
            print(f"WARNING: Stats file exists but contains invalid data: {type(normalization_stats)}")
    else:
        print(f"WARNING: Normalization stats file not found: {stats_file}")

    # Try alternative location (ds1 for any data type as fallback)
    alt_stats_file = os.path.join(exp_dir, "ds1_norm_stats.pkl")
    if os.path.exists(alt_stats_file) and alt_stats_file != stats_file:
        print(f"Trying fallback stats file: {alt_stats_file}")
        with open(alt_stats_file, "rb") as f:
            normalization_stats = pickle.load(f)

        if normalization_stats is not None and isinstance(normalization_stats, dict):
            print(f"Loaded normalization stats for keys: {list(normalization_stats.keys())}")
            return normalization_stats
        else:
            print(f"WARNING: Fallback stats file also contains invalid data: {type(normalization_stats)}")

    return None


### Checkpoint keys: ['epoch', 'global_step', 'pytorch-lightning_version', 'state_dict', 'loops', 'callbacks', 'optimizer_states', 'lr_schedulers', 'hparams_name', 'hyper_parameters']
# Hyper parameters keys: ['config_json', 'shape_meta']
# State dict sample keys: ['nets.policy.transformer.encoder.layers.0.self_attn.in_proj_weight', 'nets.policy.transformer.encoder.layers.0.self_attn.in_proj_bias', 'nets.policy.transformer.encoder.layers.0.self_attn.out_proj.weight', 'nets.policy.transformer.encoder.layers.0.self_attn.out_proj.bias', 'nets.policy.transformer.encoder.layers.0.linear1.weight', 'nets.policy.transformer.encoder.layers.0.linear1.bias', 'nets.policy.transformer.encoder.layers.0.linear2.weight', 'nets.policy.transformer.encoder.layers.0.linear2.bias', 'nets.policy.transformer.encoder.layers.0.norm1.weight', 'nets.policy.transformer.encoder.layers.0.norm1.bias']
def load_model_for_rollout(ckpt_path, data_type=0):
    """
    Load model from PyTorch Lightning checkpoint and create RolloutPolicy wrapper.

    Args:
        ckpt_path: Path to model checkpoint
        data_type: 0 for robot, 1 for hand

    Returns:
        rollout_policy: RolloutPolicy wrapper for inference
        config: Model configuration
        shape_meta: Shape metadata
    """
    import torch
    import json
    from egomimic.algo import algo_factory
    from egomimic.pl_utils.pl_data_utils import json_to_config

    print(f"Loading model from: {ckpt_path}")
    print(f"Data type: {'robot' if data_type == 0 else 'hand'}")

    # Load checkpoint directly
    ckpt = torch.load(ckpt_path, map_location='cpu')

    # Extract config and shape_meta from hyperparameters
    config_json = ckpt['hyper_parameters']['config_json']
    shape_meta = ckpt['hyper_parameters']['shape_meta']

    print("Extracted shape_meta keys:", list(shape_meta.keys()))

    # Parse the config JSON and modify it based on data_type before creating the config object
    config_dict = json.loads(config_json)

    # Extract normalization settings from config
    hdf5_normalize_obs = config_dict.get('train', {}).get('hdf5_normalize_obs', False)
    hdf5_normalize_actions = config_dict.get('train', {}).get('hdf5_normalize_actions', False)
    imagenet_normalize_images = config_dict.get('train', {}).get('imagenet_normalize_images', False)

    print(f"Normalization settings from config:")
    print(f"  hdf5_normalize_obs: {hdf5_normalize_obs}")
    print(f"  hdf5_normalize_actions: {hdf5_normalize_actions}")
    print(f"  imagenet_normalize_images: {imagenet_normalize_images}")

    # Load normalization stats if normalization was enabled during training
    normalization_stats = None
    if hdf5_normalize_obs or hdf5_normalize_actions:
        normalization_stats = load_normalization_stats(ckpt_path, data_type)
        if normalization_stats is None:
            print("WARNING: Normalization was enabled during training but stats file not found!")
            print("         Predictions may be incorrect. Check if ds1_norm_stats.pkl exists.")

    if data_type == 1 and 'observation_hand' in config_dict:
        # For hand data, replace the observation config with observation_hand
        print("Using hand observation configuration")
        print(f"Hand obs modalities: {config_dict['observation_hand']['modalities']['obs']['low_dim']}")

        # Replace observation with observation_hand in the config, but preserve encoder config
        # observation_hand only has modalities, so we need to copy encoder from original observation
        hand_observation = config_dict['observation_hand'].copy()
        hand_observation['encoder'] = config_dict['observation']['encoder']
        config_dict['observation'] = hand_observation

        # Create hand observation shapes that match ee_pose instead of joint_positions
        hand_obs_shapes = {}
        for key, shape in shape_meta["all_shapes"].items():
            if key == 'joint_positions':
                # Replace joint_positions with ee_pose for hand data
                hand_obs_shapes['ee_pose'] = [6]  # 6D pose (3 pos + 3 rot)
            else:
                # Keep other observations (like images) as-is
                hand_obs_shapes[key] = shape

        print(f"Updated obs shapes for hand: {hand_obs_shapes}")
        obs_shapes_to_use = hand_obs_shapes
    else:
        # Use robot observation configuration (default)
        print("Using robot observation configuration")
        print(f"Robot obs modalities: {config_dict['observation']['modalities']['obs']['low_dim']}")
        obs_shapes_to_use = shape_meta["all_shapes"]

    # Create config from the modified JSON
    modified_config_json = json.dumps(config_dict)
    config = json_to_config(modified_config_json)

    print(f"Algorithm: {config.algo_name}")
    print(f"Action dimension: {shape_meta.get('ac_dim', 'Unknown')}")
    print(f"All shapes: {list(obs_shapes_to_use.keys())}")

    # Create the model using the algo factory
    model = algo_factory(
        algo_name=config.algo_name,
        config=config,
        obs_key_shapes=obs_shapes_to_use,
        ac_dim=shape_meta["ac_dim"],
        device="cuda"
    )

    # Load the state dict into the model
    state_dict = ckpt['state_dict']

    # Remove the 'model.' or 'nets.' prefix from keys if present
    clean_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('nets.'):
            clean_key = k[5:]  # Remove 'nets.'
            clean_state_dict[clean_key] = v
        elif k.startswith('model.'):
            clean_key = k[6:]  # Remove 'model.'
            clean_state_dict[clean_key] = v
        else:
            clean_state_dict[k] = v

    # Load the cleaned state dict
    try:
        model.nets.load_state_dict(clean_state_dict, strict=False)
        print("Successfully loaded model weights")
    except Exception as e:
        print(f"Error loading state dict: {e}")
        # Try loading into policy specifically
        try:
            model.nets['policy'].load_state_dict(clean_state_dict, strict=False)
            print("Successfully loaded policy weights")
        except Exception as e2:
            print(f"Error loading policy weights: {e2}")

    # Set model to eval mode
    if hasattr(model, 'eval'):
        model.eval()
    for net_name, net in model.nets.items():
        if hasattr(net, 'eval'):
            net.eval()

    # Initialize observation utilities
    ObsUtils.initialize_obs_utils_with_config(config)

    # Create custom RolloutPolicy wrapper with normalization settings
    rollout_policy = EgoMimicRolloutPolicy(
        policy=model,
        obs_normalization_stats=normalization_stats,
        data_type=data_type,
        hdf5_normalize_obs=hdf5_normalize_obs,
        hdf5_normalize_actions=hdf5_normalize_actions,
        imagenet_normalize_images=imagenet_normalize_images
    )

    print("RolloutPolicy created successfully!")

    # Update shape_meta if using hand configuration
    if data_type == 1 and hasattr(config, 'observation_hand'):
        updated_shape_meta = shape_meta.copy()
        updated_shape_meta["all_shapes"] = obs_shapes_to_use
        return rollout_policy, config, updated_shape_meta

    return rollout_policy, config, shape_meta


def prepare_observation(image, config, device="cuda", imagenet_normalize=False):
    """
    Prepare observation dictionary from image for model input.
    Matches the preprocessing done during training via process_obs_dict.

    Args:
        image: Input image (H, W, 3) numpy array
        config: Model configuration
        device: Device to put tensors on
        imagenet_normalize: If True, apply ImageNet normalization (must match training config)

    Returns:
        obs_dict: Formatted observation dictionary
    """
    # Convert to float and scale to [0, 1]
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
        image = np.clip(image, 0.0, 1.0)

    # Apply ImageNet normalization if enabled (must match training config)
    if imagenet_normalize:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std

    # Convert HWC to CHW
    if len(image.shape) == 3 and image.shape[-1] == 3:
        image = np.transpose(image, (2, 0, 1))

    # Convert to tensor
    image_tensor = torch.from_numpy(image).float().unsqueeze(0)  # Add batch dimension

    if device == "cuda" and torch.cuda.is_available():
        image_tensor = image_tensor.cuda()

    # Create observation dictionary based on config
    obs_dict = {}

    # Add RGB observation
    for rgb_key in config.observation.modalities.obs.rgb:
        obs_dict[rgb_key] = image_tensor

    # Add dummy low-dim observations if needed
    for low_dim_key in config.observation.modalities.obs.low_dim:
        if low_dim_key == "joint_positions":
            # Dummy joint positions for robot data
            dummy_joints = torch.zeros(1, 14).float()
            if device == "cuda" and torch.cuda.is_available():
                dummy_joints = dummy_joints.cuda()
            obs_dict[low_dim_key] = dummy_joints
        elif low_dim_key == "ee_pose":
            # Dummy end-effector pose for hand data
            dummy_ee = torch.zeros(1, 6).float()
            if device == "cuda" and torch.cuda.is_available():
                dummy_ee = dummy_ee.cuda()
            obs_dict[low_dim_key] = dummy_ee

    return obs_dict


def get_demo_list_from_dataset(dataset_path, num_demos=3, use_val_split=False):
    """
    Get list of demo names from dataset, optionally filtering by train/val split.
    Uses the same logic as the training script to ensure consistency.
    
    Args:
        dataset_path: Path to HDF5 dataset
        num_demos: Maximum number of demos to return
        use_val_split: If True, only return validation split demos
        
    Returns:
        demo_names: List of demo names to process
        has_split_info: Dict with split information (train_mask, val_mask, has_masks)
    """
    with h5py.File(dataset_path, 'r') as f:
        # Check for train/val masks
        has_masks = 'mask' in f and 'train' in f['mask'] and 'valid' in f['mask']
        
        split_info = {
            'has_masks': has_masks,
            'train_mask': None,
            'val_mask': None
        }
        
        if has_masks:
            # Load masks exactly as training script does (from file_utils.py)
            train_mask = [elem.decode("utf-8") for elem in np.array(f['mask/train'][:])]
            val_mask = [elem.decode("utf-8") for elem in np.array(f['mask/valid'][:])]
            split_info['train_mask'] = train_mask
            split_info['val_mask'] = val_mask
            print(f"Found train/val split: {len(train_mask)} train demos, {len(val_mask)} val demos")
        else:
            print("No train/val split found in dataset")
        
        # Filter by split if requested
        if use_val_split:
            if not has_masks:
                print("WARNING: --val_split requested but no validation mask found in dataset.")
                print("Using all available demos instead.")
                # Get all demo names as fallback
                if 'data' in f:
                    all_demos = [k for k in f['data'].keys() if k.startswith('demo_')]
                else:
                    all_demos = [k for k in f.keys() if k.startswith('demo_')]
                demo_names = sorted(all_demos, key=lambda x: int(x.split('_')[1]))[:num_demos]
            else:
                # Use validation demos in the order they appear in the mask
                # This matches exactly how the training script loads them
                if num_demos > 0:
                    demo_names = split_info['val_mask'][:num_demos]
                else:
                    demo_names = split_info['val_mask']
                print(f"Using validation split: {len(demo_names)} demos (from {len(split_info['val_mask'])} total val demos)")
        else:
            # When not using val split, get all demos sorted by number
            if 'data' in f:
                all_demos = [k for k in f['data'].keys() if k.startswith('demo_')]
            else:
                all_demos = [k for k in f.keys() if k.startswith('demo_')]
            if num_demos > 0:
                demo_names = sorted(all_demos, key=lambda x: int(x.split('_')[1]))[:num_demos]
            else:
                demo_names = sorted(all_demos, key=lambda x: int(x.split('_')[1]))
        
        print(f"Processing demos: {demo_names}")
        
        return demo_names, split_info


def visualize_actions_on_dataset(rollout_policy, config, dataset_path, output_dir, num_frames, shape_meta, num_demos=3, use_val_split=False):
    """
    Run inference and create videos with predicted XYZ actions visualized on RGB frames.
    
    Args:
        rollout_policy: RolloutPolicy wrapper
        config: Model configuration  
        dataset_path: Path to HDF5 dataset
        output_dir: Output directory for videos
        num_frames: Number of frames to process per demo
        shape_meta: Shape metadata
        num_demos: Number of demos to process
        use_val_split: If True, only use validation split demos (requires mask/valid in dataset)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract task name from dataset path
    # Example: /path/to/PutBananaOnSaucer.hdf5 -> PutBananaOnSaucer
    task_name = os.path.splitext(os.path.basename(dataset_path))[0]
    print(f"\nTask name: {task_name}")
    print(f"Creating visualization videos from dataset: {dataset_path}")
    
    # Get demo list using common function
    demo_names, split_info = get_demo_list_from_dataset(dataset_path, num_demos, use_val_split)
    
    with h5py.File(dataset_path, 'r') as f:
        for demo_name in demo_names:
            if task_name == "TurnCupUpsideDown" and demo_name == "demo_277":
                # TODO: some viz issue in this demo, skip for now
                continue
            print(f"\nProcessing {demo_name} for visualization...")

            # Reset policy state for new demo
            rollout_policy.reset()

            if 'data' in f:
                demo = f['data'][demo_name]
            else:
                demo = f[demo_name]
            
            # Load data
            images = demo['obs/front_img_1'][:]  # (N, H, W, 3)
            
            # Load intrinsics and extrinsics for projection
            intrinsics = None
            extrinsics = None
            if 'obs/intrinsics' in demo and 'obs/extrinsics' in demo:
                intrinsics = demo['obs/intrinsics'][:]  # (N, 3, 3) or (3, 3)
                extrinsics = demo['obs/extrinsics'][:]  # (N, 4, 4) or (4, 4)
                print(f"Intrinsics shape: {intrinsics.shape}")
                print(f"Extrinsics shape: {extrinsics.shape}")
            else:
                print("Warning: No intrinsics/extrinsics found in dataset, using aria defaults")
                intrinsics = ARIA_INTRINSICS
                extrinsics = EXTRINSICS_LEFT
            
            # Load ground truth XYZ actions if available
            gt_actions_xyz = None
            gt_actions_joints = None
            if 'actions_xyz_act' in demo:
                gt_actions_xyz = demo['actions_xyz_act'][:]
                gt_actions_joints = demo['actions_joints_act'][:] if 'actions_joints_act' in demo else None
                print(f"Ground truth XYZ actions shape: {gt_actions_xyz.shape}")
            
            N = min(len(images), num_frames)
            print(f"Processing {N} frames...")
            
            video_frames = []
            
            for t in range(N):
                # Get observation
                image = images[t]
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                
                # Make a copy for visualization
                vis_image = image.copy()

                # Prepare observation for model (use imagenet_normalize from rollout_policy)
                obs_dict = prepare_observation(
                    image, config,
                    imagenet_normalize=rollout_policy.imagenet_normalize_images
                )

                # Add proprioceptive observations from dataset
                if 'obs' in demo:
                    obs = demo['obs']
                    expected_obs_keys = list(shape_meta.get('all_shapes', {}).keys())

                    # DEBUG: Print expected obs keys on first frame
                    if t == 0:
                        print(f"DEBUG: expected_obs_keys = {expected_obs_keys}")
                        print(f"DEBUG: obs keys in HDF5 = {list(obs.keys())}")

                    for obs_key in expected_obs_keys:
                        if obs_key == 'front_img_1':
                            continue
                        if obs_key == 'joint_positions' and 'joint_positions' in obs:
                            joint_pos = obs['joint_positions'][t]
                            obs_dict['joint_positions'] = torch.from_numpy(joint_pos).float().unsqueeze(0)
                        elif obs_key == 'ee_pose' and 'ee_pose' in obs:
                            ee_pose = obs['ee_pose'][t]
                            obs_dict['ee_pose'] = torch.from_numpy(ee_pose).float().unsqueeze(0)

                # DEBUG: Check what's in obs_dict before calling model
                if t == 0:
                    print(f"DEBUG: obs_dict keys before model = {list(obs_dict.keys())}")

                # Get prediction
                with torch.no_grad():
                    predicted_action = rollout_policy(obs_dict)
                
                # Get intrinsics and extrinsics for this frame
                if len(intrinsics.shape) == 3:
                    intrinsics_t = intrinsics[t]  # (3, 3)
                else:
                    intrinsics_t = intrinsics  # (3, 3)
                
                if len(extrinsics.shape) == 3:
                    extrinsics_t = extrinsics[t]  # (4, 4)
                else:
                    extrinsics_t = extrinsics  # (4, 4)
                
                # Convert 3x3 intrinsics to 3x4 if needed
                if intrinsics_t.shape == (3, 3):
                    intrinsics_t = np.hstack([intrinsics_t, np.zeros((3, 1))])
                
                print (f"Frame {t}: Intrinsics: {intrinsics_t}, Extrinsics: {extrinsics_t}")
                # Visualize predicted XYZ actions
                if isinstance(predicted_action, dict) and 'actions_xyz_act' in predicted_action:
                    predicted_xyz = predicted_action['actions_xyz_act']  # (100, 6)
                    predicted_joints = predicted_action['actions_joints_act'] if 'actions_joints_act' in predicted_action else None
                    
                    # Draw predicted XYZ trajectory as purple dots
                    vis_image = draw_both_actions_on_frame(
                        vis_image,
                        type="xyz",
                        color="Reds",
                        actions=predicted_xyz,
                        arm="both",
                        intrinsics=intrinsics_t,
                        extrinsics=extrinsics_t
                    )
                    # print ("pred gripper:", predicted_joints[7], predicted_joints[15])  # gripper actions
                
                # Visualize ground truth XYZ actions if available
                if gt_actions_xyz is not None:
                    gt_xyz_t = gt_actions_xyz[t]  # (100, 6)
                    # gt_joints_t = gt_actions_joints[t] if gt_actions_joints is not None else None
                    # Draw ground truth XYZ trajectory as green dots
                    vis_image = draw_both_actions_on_frame(
                        vis_image,
                        type="xyz",
                        color="Greens",
                        actions=gt_xyz_t,
                        arm="both",
                        intrinsics=intrinsics_t,
                        extrinsics=extrinsics_t
                    )
                    # print ("gt gripper:", gt_joints_t[7], gt_joints_t[15])  # gripper actions
                
                # Add text annotations
                cv2.putText(vis_image, f'Frame {t}/{N}', (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(vis_image, 'Red: Predicted XYZ', (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                cv2.putText(vis_image, 'Green: Ground Truth XYZ', (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                video_frames.append(vis_image)
            
            # Save video
            if video_frames:
                video_tensor = torch.stack([torch.from_numpy(frame) for frame in video_frames])
                video_path = os.path.join(output_dir, f'{task_name}_{demo_name}_visualization.mp4')
                print(f"Saving visualization video: {video_path}")
                write_video_safe(video_path, video_tensor, fps=30)
                print(f"Video saved successfully!")


def add_metrics(metrics, gt_actions, pred_actions):
    """
    Calculate and add metrics for a single demo trajectory.
    
    Args:
        metrics: Dictionary with keys 'paired_mse', 'final_mse', 'path_distance'
        gt_actions: Ground truth actions array of shape (seq_len, action_dim)
        pred_actions: Predicted actions array of shape (seq_len, action_dim)
    """
    # Paired MSE: MSE between consecutive frame pairs
    # Compare (gt_t, gt_t+1) with (pred_t, pred_t+1)
    paired_mse = np.mean(np.square((pred_actions - gt_actions) * 100), axis=0)
    
    # Final MSE: MSE at the final timestep
    final_mse = np.square((pred_actions[-1] - gt_actions[-1]) * 100)
    
    # Path distance: DTW distance between trajectories
    path_distance, _ = fastdtw(gt_actions, pred_actions, dist=euclidean)
    
    metrics["paired_mse"].append(paired_mse)
    metrics["final_mse"].append(final_mse)
    metrics["path_distance"].append(path_distance)



def run_inference_on_dataset(rollout_policy, config, dataset_path, output_dir, num_frames, shape_meta, num_demos=3, use_val_split=False):
    """
    Run inference on dataset samples and save results with metrics.
    
    Args:
        rollout_policy: RolloutPolicy wrapper
        config: Model configuration  
        dataset_path: Path to HDF5 dataset
        output_dir: Output directory for results
        num_frames: Number of frames to process per demo
        num_demos: Number of demos to process
        use_val_split: If True, only use validation split demos (requires mask/valid in dataset)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nRunning inference on dataset: {dataset_path}")
    
    # Get demo list using common function
    demo_names, split_info = get_demo_list_from_dataset(dataset_path, num_demos, use_val_split)
    
    # Initialize metrics storage
    metrics_xyz = {
        "paired_mse": [],  # MSE between consecutive frame pairs
        "final_mse": [],   # MSE at final timestep
        "path_distance": []  # DTW distance between trajectories
    }
    metrics_joints = {
        "paired_mse": [],
        "final_mse": [],
        "path_distance": []
    }
    
    with h5py.File(dataset_path, 'r') as f:
   
        all_results = []
        
        for demo_name in demo_names:
            print(f"\nProcessing {demo_name}...")
            
            if 'data' in f:
                demo = f['data'][demo_name]
            else:
                demo = f[demo_name]
            
            # Load data
            images = demo['obs/front_img_1'][:]  # (N, H, W, 3)
            
            # Determine ground truth actions based on what's available
            # LBM Ground truth joint actions shape: (391, 100, 14)
            # LBM Ground truth XYZ actions shape: (391, 100, 6)
            gt_actions_joints = None
            gt_actions_xyz = None
            if 'actions_joints_act' in demo:
                gt_actions_joints = demo['actions_joints_act'][:]
                print (f"Ground truth joint actions shape: {gt_actions_joints.shape}")
            if 'actions_xyz_act' in demo:
                gt_actions_xyz = demo['actions_xyz_act'][:]  
                print (f"Ground truth XYZ actions shape: {gt_actions_xyz.shape}")
            
            N = min(len(images), num_frames)
            
            demo_results = []
            
            # Collect predicted and ground truth trajectories for metrics
            pred_joints_traj = []
            pred_xyz_traj = []
            gt_joints_traj = []
            gt_xyz_traj = []
            
            for t in range(0, N, max(1, N//10)):  # Sample every 10% of frames
                print(f"  Processing frame {t}/{N}")
                
                # Get observation
                image = images[t]
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)

                # Prepare observation for model (use imagenet_normalize from rollout_policy)
                obs_dict = prepare_observation(
                    image, config,
                    imagenet_normalize=rollout_policy.imagenet_normalize_images
                )

                # Add proprioceptive observations from dataset
                # Provide observations that match the model's expected keys (based on data_type)
                if 'obs' in demo:
                    obs = demo['obs']
                    expected_obs_keys = list(shape_meta.get('all_shapes', {}).keys())
                    # Expected obs keys: ['front_img_1', 'joint_positions']
                    # Available obs keys: ['ee_pose', 'extrinsics', 'front_img_1', 'front_img_1_depth', 'intrinsics', 'joint_positions']
                    print(f"Expected obs keys: {expected_obs_keys}")
                    print(f"Available obs keys: {list(obs.keys())}")
                    
                    # Provide observations that the model expects
                    for obs_key in expected_obs_keys:
                        if obs_key == 'front_img_1':
                            continue  # Image already handled above
                        if obs_key == 'joint_positions':
                            # Robot model expects joint_positions
                            if 'joint_positions' in obs:
                                joint_pos = obs['joint_positions'][t]
                                obs_dict['joint_positions'] = torch.from_numpy(joint_pos).float().unsqueeze(0)
                                print(f"Using joint_positions: {joint_pos[:3]}...")
                            else:
                                # Fallback: create dummy joint positions if not available
                                dummy_joints = np.zeros(14, dtype=np.float32)
                                obs_dict['joint_positions'] = torch.from_numpy(dummy_joints).float().unsqueeze(0)
                                print("Using dummy joint_positions")
                        if obs_key == 'ee_pose':
                            # Hand model expects ee_pose
                            if 'ee_pose' in obs:
                                ee_pose = obs['ee_pose'][t]
                                obs_dict['ee_pose'] = torch.from_numpy(ee_pose).float().unsqueeze(0)
                                print(f"Using ee_pose: {ee_pose[:3]}...")
                            else:
                                # Fallback: create dummy ee_pose if not available
                                dummy_ee = np.zeros(6, dtype=np.float32)  # 6D pose
                                obs_dict['ee_pose'] = torch.from_numpy(dummy_ee).float().unsqueeze(0)
                                print("Using dummy ee_pose")
                
                # Get prediction
                with torch.no_grad():
                    predicted_action = rollout_policy(obs_dict)
                
                # Handle dictionary return (robot data with both joint and XYZ actions)
                if isinstance(predicted_action, dict):
                    print(f"Prediction is a dictionary with keys: {predicted_action.keys()}")
                    # For robot data, we have both joint and XYZ actions
                    predicted_joints = predicted_action['actions_joints_act']
                    predicted_xyz = predicted_action['actions_xyz_act']
                    
                    # Convert to numpy for metrics
                    if torch.is_tensor(predicted_joints):
                        predicted_joints = predicted_joints.cpu().numpy()
                    if torch.is_tensor(predicted_xyz):
                        predicted_xyz = predicted_xyz.cpu().numpy()
                    
                    # Flatten to 1D if needed (e.g., from (1, 100, 14) to (100, 14) to (14,))
                    if predicted_joints.ndim > 1:
                        predicted_joints = predicted_joints[0] if predicted_joints.shape[0] == 1 else predicted_joints
                        if predicted_joints.ndim > 1:  # Still 2D, take first timestep
                            predicted_joints = predicted_joints[0]
                    if predicted_xyz.ndim > 1:
                        predicted_xyz = predicted_xyz[0] if predicted_xyz.shape[0] == 1 else predicted_xyz
                        if predicted_xyz.ndim > 1:
                            predicted_xyz = predicted_xyz[0]
                    
                    pred_joints_traj.append(predicted_joints)
                    pred_xyz_traj.append(predicted_xyz)
                    
                    # Parse both action types
                    parsed_joints = parse_egomimic_actions(predicted_joints)
                    parsed_xyz = parse_egomimic_actions(predicted_xyz)
                    
                    # Combine parsed results
                    parsed_pred = {
                        "joint_actions": parsed_joints.get("joint_actions"),
                        "gripper_actions": parsed_joints.get("gripper_actions"),
                        "xyz_actions": parsed_xyz.get("xyz_actions"),
                        "raw_joints": predicted_joints.tolist() if hasattr(predicted_joints, 'tolist') else predicted_joints,
                        "raw_xyz": predicted_xyz.tolist() if hasattr(predicted_xyz, 'tolist') else predicted_xyz
                    }
                else:
                    # Single action array (hand data or single action type)
                    # Convert to numpy if tensor
                    if torch.is_tensor(predicted_action):
                        predicted_action = predicted_action.cpu().numpy()
                    
                    # Flatten if needed
                    if predicted_action.ndim > 1:
                        predicted_action = predicted_action[0] if predicted_action.shape[0] == 1 else predicted_action
                        if predicted_action.ndim > 1:
                            predicted_action = predicted_action[0]
                    
                    # Determine action type and add to appropriate trajectory
                    if predicted_action.shape[-1] in [14, 7]:  # Joint actions
                        pred_joints_traj.append(predicted_action)
                    elif predicted_action.shape[-1] in [6, 3]:  # XYZ actions
                        pred_xyz_traj.append(predicted_action)
                    
                    # Parse actions
                    parsed_pred = parse_egomimic_actions(predicted_action)
                
                # Get ground truth for comparison and trajectory collection
                gt_parsed = None
                if gt_actions_joints is not None:
                    if len(gt_actions_joints.shape) == 3:  # (N, seq_len, action_dim)
                        gt_action = gt_actions_joints[t, 0]  # Take first timestep
                    else:  # (N, action_dim)
                        gt_action = gt_actions_joints[t]
                    gt_joints_traj.append(gt_action)
                    gt_parsed_joints = parse_egomimic_actions(gt_action)
                    
                    # Add XYZ ground truth if available
                    if gt_actions_xyz is not None:
                        if len(gt_actions_xyz.shape) == 3:
                            gt_xyz = gt_actions_xyz[t, 0]
                        else:
                            gt_xyz = gt_actions_xyz[t]
                        gt_xyz_traj.append(gt_xyz)
                        gt_parsed_xyz = parse_egomimic_actions(gt_xyz)
                        
                        # Combine ground truth
                        gt_parsed = {
                            "joint_actions": gt_parsed_joints.get("joint_actions"),
                            "gripper_actions": gt_parsed_joints.get("gripper_actions"),
                            "xyz_actions": gt_parsed_xyz.get("xyz_actions")
                        }
                    else:
                        gt_parsed = gt_parsed_joints
                
                frame_result = {
                    "frame": t,
                    "predicted": parsed_pred,
                    "ground_truth": gt_parsed,
                }
                if not isinstance(predicted_action, dict):
                    frame_result["raw_prediction"] = predicted_action.tolist() if hasattr(predicted_action, 'tolist') else predicted_action
                
                demo_results.append(frame_result)
            
            # Compute metrics for this demo
            if len(pred_xyz_traj) > 0 and len(gt_xyz_traj) > 0:
                pred_xyz_array = np.array(pred_xyz_traj)
                gt_xyz_array = np.array(gt_xyz_traj)
                print(f"Computing XYZ metrics: pred shape {pred_xyz_array.shape}, gt shape {gt_xyz_array.shape}")
                add_metrics(metrics_xyz, gt_xyz_array, pred_xyz_array)
            
            if len(pred_joints_traj) > 0 and len(gt_joints_traj) > 0:
                pred_joints_array = np.array(pred_joints_traj)
                gt_joints_array = np.array(gt_joints_traj)
                print(f"Computing joint metrics: pred shape {pred_joints_array.shape}, gt shape {gt_joints_array.shape}")
                add_metrics(metrics_joints, gt_joints_array, pred_joints_array)
            
            all_results.append({
                "demo": demo_name,
                "results": demo_results
            })
    
    # Save results
    results_path = os.path.join(output_dir, "inference_results.json")
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Compute summary metrics
    summary = {}
    
    # XYZ metrics summary
    if len(metrics_xyz["paired_mse"]) > 0:
        print("\n=== XYZ ACTION METRICS ===")
        xyz_summary = {}
        for key in metrics_xyz:
            concat = np.stack(metrics_xyz[key], axis=0)
            mean_stat = np.mean(concat, axis=0)
            xyz_summary[key] = mean_stat
        
        # XYZ actions have 6 dimensions (left_x, left_y, left_z, right_x, right_y, right_z)
        # or 3 dimensions (x, y, z) for single arm
        if xyz_summary["paired_mse"].shape[0] == 6:
            summary.update({
                "xyz_paired_mse_left_x": float(xyz_summary["paired_mse"][0]),
                "xyz_paired_mse_left_y": float(xyz_summary["paired_mse"][1]),
                "xyz_paired_mse_left_z": float(xyz_summary["paired_mse"][2]),
                "xyz_paired_mse_right_x": float(xyz_summary["paired_mse"][3]),
                "xyz_paired_mse_right_y": float(xyz_summary["paired_mse"][4]),
                "xyz_paired_mse_right_z": float(xyz_summary["paired_mse"][5]),
                "xyz_paired_mse_avg": float(np.mean(xyz_summary["paired_mse"])),
                "xyz_final_mse_left_x": float(xyz_summary["final_mse"][0]),
                "xyz_final_mse_left_y": float(xyz_summary["final_mse"][1]),
                "xyz_final_mse_left_z": float(xyz_summary["final_mse"][2]),
                "xyz_final_mse_right_x": float(xyz_summary["final_mse"][3]),
                "xyz_final_mse_right_y": float(xyz_summary["final_mse"][4]),
                "xyz_final_mse_right_z": float(xyz_summary["final_mse"][5]),
                "xyz_final_mse_avg": float(np.mean(xyz_summary["final_mse"])),
                "xyz_path_distance_avg": float(np.mean(xyz_summary["path_distance"])),
            })
        else:  # 3D actions (single arm or averaged)
            summary.update({
                "xyz_paired_mse_x": float(xyz_summary["paired_mse"][0]),
                "xyz_paired_mse_y": float(xyz_summary["paired_mse"][1]),
                "xyz_paired_mse_z": float(xyz_summary["paired_mse"][2]),
                "xyz_paired_mse_avg": float(np.mean(xyz_summary["paired_mse"])),
                "xyz_final_mse_x": float(xyz_summary["final_mse"][0]),
                "xyz_final_mse_y": float(xyz_summary["final_mse"][1]),
                "xyz_final_mse_z": float(xyz_summary["final_mse"][2]),
                "xyz_final_mse_avg": float(np.mean(xyz_summary["final_mse"])),
                "xyz_path_distance_avg": float(np.mean(xyz_summary["path_distance"])),
            })
        
        # Print XYZ metrics
        for key, value in summary.items():
            if key.startswith("xyz_"):
                print(f"{key}: {value:.4f}")
    
    # Joint metrics summary
    if len(metrics_joints["paired_mse"]) > 0:
        print("\n=== JOINT ACTION METRICS ===")
        joints_summary = {}
        for key in metrics_joints:
            concat = np.stack(metrics_joints[key], axis=0)
            mean_stat = np.mean(concat, axis=0)
            joints_summary[key] = mean_stat
        
        # Joint actions are averaged across all dimensions
        summary.update({
            "joints_paired_mse_avg": float(np.mean(joints_summary["paired_mse"])),
            "joints_final_mse_avg": float(np.mean(joints_summary["final_mse"])),
            "joints_path_distance_avg": float(np.mean(joints_summary["path_distance"])),
        })
        
        # Print joint metrics
        for key, value in summary.items():
            if key.startswith("joints_"):
                print(f"{key}: {value:.4f}")
    
    # Save metrics summary
    metrics_path = os.path.join(output_dir, "metrics_summary.json")
    with open(metrics_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nMetrics saved to: {metrics_path}")
    
    print(f"\nResults saved to: {results_path}")


def run_interactive_inference(rollout_policy, config):
    """
    Run interactive inference where you can provide images.
    
    Args:
        rollout_policy: RolloutPolicy wrapper
        config: Model configuration
    """
    print("\n=== INTERACTIVE INFERENCE MODE ===")
    print("Enter image path (or 'quit' to exit):")
    
    while True:
        image_path = input("Image path: ").strip()
        
        if image_path.lower() in ['quit', 'exit', 'q']:
            break
            
        if not os.path.exists(image_path):
            print(f"File not found: {image_path}")
            continue
            
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Could not load image: {image_path}")
                continue
                
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Prepare observation (use imagenet_normalize from rollout_policy)
            obs_dict = prepare_observation(
                image, config,
                imagenet_normalize=rollout_policy.imagenet_normalize_images
            )

            # Get prediction
            with torch.no_grad():
                predicted_action = rollout_policy(obs_dict)
            
            # Handle dictionary return (robot data with both joint and XYZ actions)
            if isinstance(predicted_action, dict):
                print("\n--- PREDICTION RESULTS (Robot Data) ---")
                print("Joint Actions:")
                parsed_joints = parse_egomimic_actions(predicted_action['actions_joints_act'])
                if parsed_joints['joint_actions'] is not None:
                    print(f"  Left arm:  {parsed_joints['joint_actions']['left']}")
                    print(f"  Right arm: {parsed_joints['joint_actions']['right']}")
                if parsed_joints['gripper_actions'] is not None:
                    print(f"  Grippers:  {parsed_joints['gripper_actions']['combined']}")
                
                print("\nXYZ Actions:")
                parsed_xyz = parse_egomimic_actions(predicted_action['actions_xyz_act'])
                if parsed_xyz['xyz_actions'] is not None:
                    print(f"  Left arm:  {parsed_xyz['xyz_actions']['left']}")
                    print(f"  Right arm: {parsed_xyz['xyz_actions']['right']}")
                print()
            else:
                # Single action array (hand data)
                # Convert to numpy if tensor
                if torch.is_tensor(predicted_action):
                    predicted_action = predicted_action.cpu().numpy()
                
                # Parse and display results
                parsed = parse_egomimic_actions(predicted_action)
                
                print("\n--- PREDICTION RESULTS ---")
                if parsed['joint_actions'] is not None:
                    print(f"Joint Actions (Left arm):  {parsed['joint_actions']['left']}")
                    print(f"Joint Actions (Right arm): {parsed['joint_actions']['right']}")
                if parsed['xyz_actions'] is not None:
                    print(f"XYZ Actions (Left arm):    {parsed['xyz_actions']['left']}")
                    print(f"XYZ Actions (Right arm):   {parsed['xyz_actions']['right']}")
                if parsed['gripper_actions'] is not None:
                    print(f"Gripper Actions:           {parsed['gripper_actions']['combined']}")
                print()
            
        except Exception as e:
            print(f"Error processing image: {e}")


def run_inference_on_directory(rollout_policy, config, dataset_dir, output_dir, num_frames, shape_meta, num_demos=3, use_val_split=False, visualize=False):
    """
    Run inference on all HDF5 files in a directory and aggregate metrics across all tasks.
    
    Args:
        rollout_policy: RolloutPolicy wrapper
        config: Model configuration
        dataset_dir: Directory containing HDF5 dataset files
        output_dir: Output directory for results
        num_frames: Number of frames to process per demo
        shape_meta: Shape metadata
        num_demos: Number of demos to process per task
        use_val_split: If True, only use validation split demos
        visualize: If True, generate visualization videos
    """
    import glob
    
    # Find all HDF5 files in the directory
    hdf5_files = glob.glob(os.path.join(dataset_dir, "*.hdf5"))
    
    if not hdf5_files:
        print(f"No HDF5 files found in directory: {dataset_dir}")
        return
    
    print(f"\nFound {len(hdf5_files)} HDF5 files in {dataset_dir}")
    print(f"Tasks: {[os.path.basename(f) for f in hdf5_files]}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Aggregate metrics across all tasks
    all_tasks_metrics = {
        "tasks": [],
        "aggregated_xyz": {
            "paired_mse": [],
            "final_mse": [],
            "path_distance": []
        },
        "aggregated_joints": {
            "paired_mse": [],
            "final_mse": [],
            "path_distance": []
        }
    }
    
    # Process each HDF5 file
    for hdf5_file in sorted(hdf5_files):
        task_name = os.path.splitext(os.path.basename(hdf5_file))[0]
        print(f"\n{'='*80}")
        print(f"Processing task: {task_name}")
        print(f"{'='*80}")
        
        # Create task-specific output directory
        task_output_dir = os.path.join(output_dir, task_name)
        os.makedirs(task_output_dir, exist_ok=True)
        
        try:
            if visualize:
                # Run visualization for this task
                visualize_actions_on_dataset(
                    rollout_policy,
                    config,
                    hdf5_file,
                    task_output_dir,
                    num_frames,
                    shape_meta,
                    num_demos,
                    use_val_split
                )
            else:
                # Run inference and get metrics for this task
                task_metrics = run_inference_on_dataset_with_metrics(
                    rollout_policy,
                    config,
                    hdf5_file,
                    task_output_dir,
                    num_frames,
                    shape_meta,
                    num_demos,
                    use_val_split
                )
                
                # Store task metrics (only summary, not the raw arrays for JSON serialization)
                all_tasks_metrics["tasks"].append({
                    "task_name": task_name,
                    "summary": task_metrics["summary"]  # Only store the summary dict, not the raw metrics
                })
                
                # Aggregate metrics across tasks (use numpy arrays for computation)
                if "xyz_metrics" in task_metrics and task_metrics["xyz_metrics"]:
                    xyz_metrics = task_metrics["xyz_metrics"]
                    all_tasks_metrics["aggregated_xyz"]["paired_mse"].extend(xyz_metrics["paired_mse"])
                    all_tasks_metrics["aggregated_xyz"]["final_mse"].extend(xyz_metrics["final_mse"])
                    all_tasks_metrics["aggregated_xyz"]["path_distance"].extend(xyz_metrics["path_distance"])
                
                if "joints_metrics" in task_metrics and task_metrics["joints_metrics"]:
                    joints_metrics = task_metrics["joints_metrics"]
                    all_tasks_metrics["aggregated_joints"]["paired_mse"].extend(joints_metrics["paired_mse"])
                    all_tasks_metrics["aggregated_joints"]["final_mse"].extend(joints_metrics["final_mse"])
                    all_tasks_metrics["aggregated_joints"]["path_distance"].extend(joints_metrics["path_distance"])
        
        except Exception as e:
            print(f"Error processing {task_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Compute and save aggregated metrics across all tasks
    if not visualize:
        print(f"\n{'='*80}")
        print("AGGREGATED METRICS ACROSS ALL TASKS")
        print(f"{'='*80}")
        
        aggregated_summary = {}
        
        # XYZ metrics
        if len(all_tasks_metrics["aggregated_xyz"]["paired_mse"]) > 0:
            print("\n=== AGGREGATED XYZ ACTION METRICS ===")
            xyz_paired_mse = np.stack(all_tasks_metrics["aggregated_xyz"]["paired_mse"], axis=0)
            xyz_final_mse = np.stack(all_tasks_metrics["aggregated_xyz"]["final_mse"], axis=0)
            xyz_path_distance = all_tasks_metrics["aggregated_xyz"]["path_distance"]
            
            mean_paired_mse = np.mean(xyz_paired_mse, axis=0)
            mean_final_mse = np.mean(xyz_final_mse, axis=0)
            mean_path_distance = np.mean(xyz_path_distance)
            
            if mean_paired_mse.shape[0] == 6:
                aggregated_summary.update({
                    "xyz_paired_mse_left_x": float(mean_paired_mse[0]),
                    "xyz_paired_mse_left_y": float(mean_paired_mse[1]),
                    "xyz_paired_mse_left_z": float(mean_paired_mse[2]),
                    "xyz_paired_mse_right_x": float(mean_paired_mse[3]),
                    "xyz_paired_mse_right_y": float(mean_paired_mse[4]),
                    "xyz_paired_mse_right_z": float(mean_paired_mse[5]),
                    "xyz_paired_mse_avg": float(np.mean(mean_paired_mse)),
                    "xyz_final_mse_left_x": float(mean_final_mse[0]),
                    "xyz_final_mse_left_y": float(mean_final_mse[1]),
                    "xyz_final_mse_left_z": float(mean_final_mse[2]),
                    "xyz_final_mse_right_x": float(mean_final_mse[3]),
                    "xyz_final_mse_right_y": float(mean_final_mse[4]),
                    "xyz_final_mse_right_z": float(mean_final_mse[5]),
                    "xyz_final_mse_avg": float(np.mean(mean_final_mse)),
                    "xyz_path_distance_avg": float(mean_path_distance),
                })
            else:
                aggregated_summary.update({
                    "xyz_paired_mse_x": float(mean_paired_mse[0]),
                    "xyz_paired_mse_y": float(mean_paired_mse[1]),
                    "xyz_paired_mse_z": float(mean_paired_mse[2]),
                    "xyz_paired_mse_avg": float(np.mean(mean_paired_mse)),
                    "xyz_final_mse_x": float(mean_final_mse[0]),
                    "xyz_final_mse_y": float(mean_final_mse[1]),
                    "xyz_final_mse_z": float(mean_final_mse[2]),
                    "xyz_final_mse_avg": float(np.mean(mean_final_mse)),
                    "xyz_path_distance_avg": float(mean_path_distance),
                })
            
            for key, value in aggregated_summary.items():
                if key.startswith("xyz_"):
                    print(f"{key}: {value:.4f}")
        
        # Joint metrics
        if len(all_tasks_metrics["aggregated_joints"]["paired_mse"]) > 0:
            print("\n=== AGGREGATED JOINT ACTION METRICS ===")
            joints_paired_mse = np.stack(all_tasks_metrics["aggregated_joints"]["paired_mse"], axis=0)
            joints_final_mse = np.stack(all_tasks_metrics["aggregated_joints"]["final_mse"], axis=0)
            joints_path_distance = all_tasks_metrics["aggregated_joints"]["path_distance"]
            
            aggregated_summary.update({
                "joints_paired_mse_avg": float(np.mean(joints_paired_mse)),
                "joints_final_mse_avg": float(np.mean(joints_final_mse)),
                "joints_path_distance_avg": float(np.mean(joints_path_distance)),
            })
            
            for key, value in aggregated_summary.items():
                if key.startswith("joints_"):
                    print(f"{key}: {value:.4f}")
        
        # Save aggregated metrics
        aggregated_path = os.path.join(output_dir, "aggregated_metrics_all_tasks.json")
        with open(aggregated_path, 'w') as f:
            json.dump({
                "aggregated_summary": aggregated_summary,
                "per_task_metrics": all_tasks_metrics["tasks"],
                "num_tasks": len(hdf5_files),
                "num_demos_per_task": num_demos
            }, f, indent=2)
        print(f"\nAggregated metrics saved to: {aggregated_path}")


def run_inference_on_dataset_with_metrics(rollout_policy, config, dataset_path, output_dir, num_frames, shape_meta, num_demos=3, use_val_split=False):
    """
    Run inference on dataset and return metrics (for aggregation across tasks).
    
    Returns:
        dict: Contains xyz_metrics and joints_metrics for this task
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nRunning inference on dataset: {dataset_path}")
    
    # Get demo list using common function
    demo_names, split_info = get_demo_list_from_dataset(dataset_path, num_demos, use_val_split)
    
    # Initialize metrics storage
    metrics_xyz = {
        "paired_mse": [],
        "final_mse": [],
        "path_distance": []
    }
    metrics_joints = {
        "paired_mse": [],
        "final_mse": [],
        "path_distance": []
    }
    
    with h5py.File(dataset_path, 'r') as f:
        all_results = []
        
        for demo_name in demo_names:
            print(f"\nProcessing {demo_name}...")
            
            if 'data' in f:
                demo = f['data'][demo_name]
            else:
                demo = f[demo_name]
            
            # Load data
            images = demo['obs/front_img_1'][:]
            
            gt_actions_joints = None
            gt_actions_xyz = None
            if 'actions_joints_act' in demo:
                gt_actions_joints = demo['actions_joints_act'][:]
                print(f"Ground truth joint actions shape: {gt_actions_joints.shape}")
            if 'actions_xyz_act' in demo:
                gt_actions_xyz = demo['actions_xyz_act'][:]  
                print(f"Ground truth XYZ actions shape: {gt_actions_xyz.shape}")
            
            N = min(len(images), num_frames)
            
            demo_results = []
            pred_joints_traj = []
            pred_xyz_traj = []
            gt_joints_traj = []
            gt_xyz_traj = []
            
            for t in range(0, N, max(1, N//10)):
                # Get observation
                image = images[t]
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)

                # Prepare observation (use imagenet_normalize from rollout_policy if applicable)
                obs_dict = prepare_observation(
                    image, config,
                    imagenet_normalize=rollout_policy.imagenet_normalize_images
                )

                # Add proprioceptive observations
                if 'obs' in demo:
                    obs = demo['obs']
                    expected_obs_keys = list(shape_meta.get('all_shapes', {}).keys())
                    
                    for obs_key in expected_obs_keys:
                        if obs_key == 'front_img_1':
                            continue
                        if obs_key == 'joint_positions' and 'joint_positions' in obs:
                            joint_pos = obs['joint_positions'][t]
                            obs_dict['joint_positions'] = torch.from_numpy(joint_pos).float().unsqueeze(0)
                        elif obs_key == 'ee_pose' and 'ee_pose' in obs:
                            ee_pose = obs['ee_pose'][t]
                            obs_dict['ee_pose'] = torch.from_numpy(ee_pose).float().unsqueeze(0)
                
                # Get prediction
                with torch.no_grad():
                    predicted_action = rollout_policy(obs_dict)
                
                # Process predictions
                if isinstance(predicted_action, dict):
                    predicted_joints = predicted_action['actions_joints_act']
                    predicted_xyz = predicted_action['actions_xyz_act']
                    
                    if torch.is_tensor(predicted_joints):
                        predicted_joints = predicted_joints.cpu().numpy()
                    if torch.is_tensor(predicted_xyz):
                        predicted_xyz = predicted_xyz.cpu().numpy()
                    
                    if predicted_joints.ndim > 1:
                        predicted_joints = predicted_joints[0] if predicted_joints.shape[0] == 1 else predicted_joints
                        if predicted_joints.ndim > 1:
                            predicted_joints = predicted_joints[0]
                    if predicted_xyz.ndim > 1:
                        predicted_xyz = predicted_xyz[0] if predicted_xyz.shape[0] == 1 else predicted_xyz
                        if predicted_xyz.ndim > 1:
                            predicted_xyz = predicted_xyz[0]
                    
                    pred_joints_traj.append(predicted_joints)
                    pred_xyz_traj.append(predicted_xyz)
                else:
                    if torch.is_tensor(predicted_action):
                        predicted_action = predicted_action.cpu().numpy()
                    
                    if predicted_action.ndim > 1:
                        predicted_action = predicted_action[0] if predicted_action.shape[0] == 1 else predicted_action
                        if predicted_action.ndim > 1:
                            predicted_action = predicted_action[0]
                    
                    if predicted_action.shape[-1] in [14, 7]:
                        pred_joints_traj.append(predicted_action)
                    elif predicted_action.shape[-1] in [6, 3]:
                        pred_xyz_traj.append(predicted_action)
                
                # Collect ground truth
                if gt_actions_joints is not None:
                    if len(gt_actions_joints.shape) == 3:
                        gt_action = gt_actions_joints[t, 0]
                    else:
                        gt_action = gt_actions_joints[t]
                    gt_joints_traj.append(gt_action)
                    
                    if gt_actions_xyz is not None:
                        if len(gt_actions_xyz.shape) == 3:
                            gt_xyz = gt_actions_xyz[t, 0]
                        else:
                            gt_xyz = gt_actions_xyz[t]
                        gt_xyz_traj.append(gt_xyz)
            
            # Compute metrics for this demo
            if len(pred_xyz_traj) > 0 and len(gt_xyz_traj) > 0:
                pred_xyz_array = np.array(pred_xyz_traj)
                gt_xyz_array = np.array(gt_xyz_traj)
                add_metrics(metrics_xyz, gt_xyz_array, pred_xyz_array)
            
            if len(pred_joints_traj) > 0 and len(gt_joints_traj) > 0:
                pred_joints_array = np.array(pred_joints_traj)
                gt_joints_array = np.array(gt_joints_traj)
                add_metrics(metrics_joints, gt_joints_array, pred_joints_array)
    
    # Compute summary for this task
    task_summary = {}
    
    if len(metrics_xyz["paired_mse"]) > 0:
        xyz_summary = {}
        for key in metrics_xyz:
            concat = np.stack(metrics_xyz[key], axis=0)
            mean_stat = np.mean(concat, axis=0)
            xyz_summary[key] = mean_stat
        
        task_summary["xyz_paired_mse_avg"] = float(np.mean(xyz_summary["paired_mse"]))
        task_summary["xyz_final_mse_avg"] = float(np.mean(xyz_summary["final_mse"]))
        task_summary["xyz_path_distance_avg"] = float(np.mean(xyz_summary["path_distance"]))
    
    if len(metrics_joints["paired_mse"]) > 0:
        joints_summary = {}
        for key in metrics_joints:
            concat = np.stack(metrics_joints[key], axis=0)
            mean_stat = np.mean(concat, axis=0)
            joints_summary[key] = mean_stat
        
        task_summary["joints_paired_mse_avg"] = float(np.mean(joints_summary["paired_mse"]))
        task_summary["joints_final_mse_avg"] = float(np.mean(joints_summary["final_mse"]))
        task_summary["joints_path_distance_avg"] = float(np.mean(joints_summary["path_distance"]))
    
    # Save task-specific metrics
    metrics_path = os.path.join(output_dir, "metrics_summary.json")
    with open(metrics_path, 'w') as f:
        json.dump(task_summary, f, indent=2)
    
    print(f"Task metrics: {task_summary}")
    
    # Convert numpy arrays to lists for JSON serialization
    def convert_metrics_to_lists(metrics):
        """Convert numpy arrays in metrics to Python lists."""
        converted = {}
        for key, value_list in metrics.items():
            converted[key] = [v.tolist() if isinstance(v, np.ndarray) else v for v in value_list]
        return converted
    
    # Return metrics for aggregation (with numpy arrays converted to lists)
    xyz_metrics_serializable = convert_metrics_to_lists(metrics_xyz) if len(metrics_xyz["paired_mse"]) > 0 else None
    joints_metrics_serializable = convert_metrics_to_lists(metrics_joints) if len(metrics_joints["paired_mse"]) > 0 else None
    
    return {
        "xyz_metrics": metrics_xyz if len(metrics_xyz["paired_mse"]) > 0 else None,  # Keep as numpy for aggregation
        "joints_metrics": metrics_joints if len(metrics_joints["paired_mse"]) > 0 else None,  # Keep as numpy for aggregation
        "xyz_metrics_serializable": xyz_metrics_serializable,  # For JSON serialization
        "joints_metrics_serializable": joints_metrics_serializable,  # For JSON serialization
        "summary": task_summary
    }


def main():
    parser = argparse.ArgumentParser(description='EgoMimic Inference with RolloutPolicy')
    parser.add_argument('--ckpt_path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--dataset_path', type=str, default=None,
                        help='Path to HDF5 dataset file or directory containing HDF5 files')
    parser.add_argument('--output_dir', type=str, default='./inference_output',
                        help='Output directory for results')
    parser.add_argument('--num_frames', type=int, default=200,
                        help='Number of frames to process per demo from dataset')
    parser.add_argument('--num_demos', type=int, default=3,
                        help='Number of demos to process from dataset')
    parser.add_argument('--interactive', action='store_true',
                        help='Run in interactive mode for single image inference')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate videos with visualized XYZ actions projected on RGB frames')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run inference on')
    parser.add_argument('--data_type', type=int, default=0, choices=[0, 1],
                        help='Data type: 0 = robot/joint data (LBM), 1 = hand/human data (EgoDx)')
    parser.add_argument('--val_split', action='store_true',
                        help='Only use validation split demos (requires mask/valid in dataset)')
    
    args = parser.parse_args()

    print ("num demos: ", args.num_demos)
    
    # Set device
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'
    
    # Load model
    rollout_policy, config, shape_meta = load_model_for_rollout(args.ckpt_path, args.data_type)
    
    if args.dataset_path:
        # Check if dataset_path is a directory or a file
        if os.path.isdir(args.dataset_path):
            print(f"\nDetected directory input: {args.dataset_path}")
            print("Running inference on all HDF5 files in directory...")
            
            # Process entire directory
            run_inference_on_directory(
                rollout_policy,
                config,
                args.dataset_path,
                args.output_dir,
                args.num_frames,
                shape_meta,
                args.num_demos,
                args.val_split,
                args.visualize
            )
        else:
            print(f"\nDetected single file input: {args.dataset_path}")
            
            if args.visualize:
                # Generate visualization videos with projected actions
                visualize_actions_on_dataset(
                    rollout_policy,
                    config,
                    args.dataset_path,
                    args.output_dir,
                    args.num_frames,
                    shape_meta,
                    args.num_demos,
                    args.val_split
                )
            else:
                # Run inference on single dataset (JSON output only)
                run_inference_on_dataset(
                    rollout_policy, 
                    config, 
                    args.dataset_path, 
                    args.output_dir, 
                    args.num_frames,
                    shape_meta,
                    args.num_demos,
                    args.val_split
                )
    
    if args.interactive:
        # Run interactive mode
        run_interactive_inference(rollout_policy, config)
    
    print("\nInference complete!")


if __name__ == "__main__":
    main()