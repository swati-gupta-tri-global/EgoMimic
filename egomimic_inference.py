#!/usr/bin/env python3
r"""
EgoMimic Inference Script with RolloutPolicy

This script loads a trained EgoMimic model and performs rollouts to get action predictions
from visual observations. It creates the RolloutPolicy wrapper and demonstrates how to
extract joint actions, XYZ actions, and gripper commands.

Usage:
docker exec -it swati-egomimic python /workspace/externals/EgoMimic/egomimic_inference.py --ckpt_path /workspace/externals/EgoMimic/trained_models_highlevel/test/None_DT_2025-11-22-18-26-12/models/model_epoch_epoch=139.ckpt \ 
    --dataset_path /workspace/externals/EgoMimic/datasets/egodex/processed/part1/clean_cups.hdf5 \ 
    --output_dir /workspace/externals/EgoMimic/inference_output_egodx --num_samples 1 --data_type 1

python egomimic_inference.py --ckpt_path /workspace/externals/EgoMimic/trained_models_highlevel/test/None_DT_2026-01-12-18-04-24/models/model_epoch_epoch=49.ckpt  --dataset_path datasets/LBM_sim_egocentric/train_split/PutBananaOnSaucer.hdf5 --output_dir inference_output_lbm --num_frames 200 --data_type 0 --visualize

docker exec -it swati-egomimic python /workspace/externals/EgoMimic/egomimic_inference.py --ckpt_path /workspace/externals/EgoMimic/trained_models_highlevel/test/None_DT_2026-01-06-22-03-51/models/model_epoch_epoch=19.ckpt     --dataset_path /workspace/externals/EgoMimic/datasets/LBM_sim_egocentric/processed/BimanualHangMugsOnMugHolderFromDryingRack.hdf5     --output_dir /workspace/externals/EgoMimic/inference_output --num_samples 1 --data_type 0
docker exec -it swati-egomimic python /workspace/externals/EgoMimic/egomimic_inference.py --ckpt_path /workspace/externals/EgoMimic/trained_models_highlevel/test/None_DT_2026-01-06-22-03-51/models/model_epoch_epoch=19.ckpt \ 
    --dataset_path /workspace/externals/EgoMimic/datasets/egodex/processed/part1/clean_cups.hdf5 \ 
    --output_dir /workspace/externals/EgoMimic/inference_output_egodex --num_samples 1 --data_type 1

"""

import argparse
import json
import os
import numpy as np
import torch
import h5py
import cv2
# from pathlib import Path

# Import robomimic modules
# import robomimic.utils.file_utils as FileUtils
# import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils
# from robomimic.algo import RolloutPolicy
# from robomimic.utils.file_utils import policy_from_checkpoint

# Import EgoMimic modules  
# from egomimic.configs import config_factory
from egomimic.utils.val_utils import draw_both_actions_on_frame, write_video_safe


class EgoMimicRolloutPolicy:
    """
    Custom policy wrapper for EgoMimic that implements get_action using forward_eval
    """
    def __init__(self, policy, obs_normalization_stats=None, data_type=0):
        self.policy = policy
        self.obs_normalization_stats = obs_normalization_stats
        self.data_type = data_type  # 0 = robot, 1 = hand
        self._step_counter = 0
        
    def __call__(self, obs_dict):
        return self.get_action(obs_dict)
    
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
        print(f"[DEBUG] Batch type tensor: {batch['type']}")
        # pad_mask needs to be (batch_size, sequence_length) - using 100 as sequence length
        batch["obs"]["pad_mask"] = torch.ones((1, 100), dtype=torch.float32)  # Not padded
                
        # Move to device if available
        device = next(self.policy.nets['policy'].parameters()).device
        batch["type"] = batch["type"].to(device)
        for key in batch["obs"]:
            if hasattr(batch["obs"][key], 'to'):
                batch["obs"][key] = batch["obs"][key].to(device)
        
        # Run forward_eval
        with torch.no_grad():
            predictions = self.policy.forward_eval(batch, self.obs_normalization_stats)
            
        # Debug: Print available prediction keys
        # Available prediction keys: ['actions_joints_act', 'actions_xyz_act']
        print(f"Available prediction keys: {list(predictions.keys())}")
        print(f"Expected action key: {self.policy.ac_key}")
            
        # For robot data, return both joint and XYZ actions if both are available
        if 'actions_joints_act' in predictions and 'actions_xyz_act' in predictions:
            print("Robot data: Returning both joint and XYZ actions")
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
    elif action_dim == 22:
        # Joint + Gripper + XYZ actions (7+1 per arm + 3 per arm)
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
            "xyz_actions": {
                "left": action[16:19].tolist(),
                "right": action[19:22].tolist(),
                "combined": action[16:22].tolist()
            }
        }
    else:
        print(f"[WARNING] Unexpected action dimension: {action_dim}")
        return {
            "joint_actions": None,
            "xyz_actions": None, 
            "gripper_actions": None,
            "raw_action": action.tolist() if hasattr(action, 'tolist') else action
        }

### Checkpoint keys: ['epoch', 'global_step', 'pytorch-lightning_version', 'state_dict', 'loops', 'callbacks', 'optimizer_states', 'lr_schedulers', 'hparams_name', 'hyper_parameters']
# Hyper parameters keys: ['config_json', 'shape_meta']
# State dict sample keys: ['nets.policy.transformer.encoder.layers.0.self_attn.in_proj_weight', 'nets.policy.transformer.encoder.layers.0.self_attn.in_proj_bias', 'nets.policy.transformer.encoder.layers.0.self_attn.out_proj.weight', 'nets.policy.transformer.encoder.layers.0.self_attn.out_proj.bias', 'nets.policy.transformer.encoder.layers.0.linear1.weight', 'nets.policy.transformer.encoder.layers.0.linear1.bias', 'nets.policy.transformer.encoder.layers.0.linear2.weight', 'nets.policy.transformer.encoder.layers.0.linear2.bias', 'nets.policy.transformer.encoder.layers.0.norm1.weight', 'nets.policy.transformer.encoder.layers.0.norm1.bias']
def load_model_for_rollout(ckpt_path, data_type=0):
    """
    Load model from PyTorch Lightning checkpoint and create RolloutPolicy wrapper.
    
    Args:
        ckpt_path: Path to model checkpoint
        
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

    print ("Extracted shape_meta keys:", list(shape_meta.keys()))
    
    # Parse the config JSON and modify it based on data_type before creating the config object
    config_dict = json.loads(config_json)
    
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
    
    # Create custom RolloutPolicy wrapper
    rollout_policy = EgoMimicRolloutPolicy(
        policy=model,
        obs_normalization_stats=None,
        data_type=data_type
    )
    
    print("RolloutPolicy created successfully!")
    
    # Update shape_meta if using hand configuration
    if data_type == 1 and hasattr(config, 'observation_hand'):
        updated_shape_meta = shape_meta.copy()
        updated_shape_meta["all_shapes"] = obs_shapes_to_use
        return rollout_policy, config, updated_shape_meta
    
    return rollout_policy, config, shape_meta


def prepare_observation(image, config, device="cuda"):
    """
    Prepare observation dictionary from image for model input.
    
    Args:
        image: Input image (H, W, 3) numpy array
        config: Model configuration
        device: Device to put tensors on
        
    Returns:
        obs_dict: Formatted observation dictionary
    """
    # Convert to tensor and normalize
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
    
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


def visualize_actions_on_dataset(rollout_policy, config, dataset_path, output_dir, num_frames, shape_meta, max_demos=3):
    """
    Run inference and create videos with predicted XYZ actions visualized on RGB frames.
    
    Args:
        rollout_policy: RolloutPolicy wrapper
        config: Model configuration  
        dataset_path: Path to HDF5 dataset
        output_dir: Output directory for videos
        num_samples: Number of frames to process per demo
        shape_meta: Shape metadata
        max_demos: Maximum number of demos to process
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nCreating visualization videos from dataset: {dataset_path}")
    
    with h5py.File(dataset_path, 'r') as f:
        # Get demo names
        if 'data' in f:
            demo_names = [k for k in f['data'].keys() if k.startswith('demo_')]
        else:
            demo_names = [k for k in f.keys() if k.startswith('demo_')]
        
        demo_names = sorted(demo_names)[:min(len(demo_names), max_demos)]
        
        for demo_name in demo_names:
            print(f"\nProcessing {demo_name} for visualization...")
            
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
                print("Warning: No intrinsics/extrinsics found in dataset, skipping visualization")
                continue
            
            # Load ground truth XYZ actions if available
            gt_actions_xyz = None
            if 'actions_xyz_act' in demo:
                gt_actions_xyz = demo['actions_xyz_act'][:]
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
                
                # Prepare observation for model
                obs_dict = prepare_observation(image, config)
                
                # Add proprioceptive observations from dataset
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
                
                # Visualize ground truth XYZ actions if available
                if gt_actions_xyz is not None:
                    gt_xyz_t = gt_actions_xyz[t]  # (100, 6)
                    
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
                video_path = os.path.join(output_dir, f'{demo_name}_visualization.mp4')
                print(f"Saving visualization video: {video_path}")
                write_video_safe(video_path, video_tensor, fps=10)
                print(f"Video saved successfully!")


def run_inference_on_dataset(rollout_policy, config, dataset_path, output_dir, num_frames, shape_meta, num_demos=3):
    """
    Run inference on dataset samples and save results.
    
    Args:
        rollout_policy: RolloutPolicy wrapper
        config: Model configuration  
        dataset_path: Path to HDF5 dataset
        output_dir: Output directory for results
        num_frames: Number of frames to process per demo
        num_demos: Number of demos to process
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nRunning inference on dataset: {dataset_path}")
    
    with h5py.File(dataset_path, 'r') as f:
        # Get demo names
        if 'data' in f:
            demo_names = [k for k in f['data'].keys() if k.startswith('demo_')]
        else:
            demo_names = [k for k in f.keys() if k.startswith('demo_')]
        
        demo_names = sorted(demo_names)[:min(len(demo_names), num_demos)]  # Process max num_demos demos
        
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
            
            for t in range(0, N, max(1, N//10)):  # Sample every 10% of frames
                print(f"  Processing frame {t}/{N}")
                
                # Get observation
                image = images[t]
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                
                # Prepare observation for model
                obs_dict = prepare_observation(image, config)
                
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
                    
                    # Parse actions
                    parsed_pred = parse_egomimic_actions(predicted_action)
                
                # Get ground truth for comparison
                gt_parsed = None
                if gt_actions_joints is not None:
                    if len(gt_actions_joints.shape) == 3:  # (N, seq_len, action_dim)
                        gt_action = gt_actions_joints[t, 0]  # Take first timestep
                    else:  # (N, action_dim)
                        gt_action = gt_actions_joints[t]
                    gt_parsed_joints = parse_egomimic_actions(gt_action)
                    
                    # Add XYZ ground truth if available
                    if gt_actions_xyz is not None:
                        if len(gt_actions_xyz.shape) == 3:
                            gt_xyz = gt_actions_xyz[t, 0]
                        else:
                            gt_xyz = gt_actions_xyz[t]
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
            
            all_results.append({
                "demo": demo_name,
                "results": demo_results
            })
    
    # Save results
    results_path = os.path.join(output_dir, "inference_results.json")
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to: {results_path}")
    
    # Print summary
    print("\n=== INFERENCE SUMMARY ===")
    for demo_result in all_results:
        print(f"\nDemo: {demo_result['demo']}")
        for i, frame_result in enumerate(demo_result['results'][:3]):  # Show first 3 frames
            print(f"  Frame {frame_result['frame']}:")
            pred = frame_result['predicted']
            if pred['joint_actions'] is not None:
                print(f"    Predicted joints (left): {pred['joint_actions']['left'][:3]}...")
                print(f"    Predicted joints (right): {pred['joint_actions']['right'][:3]}...")
            if pred['xyz_actions'] is not None:
                print(f"    Predicted XYZ (left): {pred['xyz_actions']['left']}")
                print(f"    Predicted XYZ (right): {pred['xyz_actions']['right']}")


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
            
            # Prepare observation
            obs_dict = prepare_observation(image, config)
            
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


def main():
    parser = argparse.ArgumentParser(description='EgoMimic Inference with RolloutPolicy')
    parser.add_argument('--ckpt_path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--dataset_path', type=str, default=None,
                        help='Path to HDF5 dataset for inference (optional)')
    parser.add_argument('--output_dir', type=str, default='./inference_output',
                        help='Output directory for results')
    parser.add_argument('--num_frames', type=int, default=10,
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
    
    args = parser.parse_args()
    
    # Set device
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'
    
    # Load model
    rollout_policy, config, shape_meta = load_model_for_rollout(args.ckpt_path, args.data_type)
    
    if args.dataset_path:
        if args.visualize:
            # Generate visualization videos with projected actions
            visualize_actions_on_dataset(
                rollout_policy,
                config,
                args.dataset_path,
                args.output_dir,
                args.num_frames,
                shape_meta,
                max_demos=args.num_demos
            )
        else:
            # Run inference on dataset (JSON output only)
            run_inference_on_dataset(
                rollout_policy, 
                config, 
                args.dataset_path, 
                args.output_dir, 
                args.num_frames,
                shape_meta,
                max_demos=args.num_demos
            )
    
    if args.interactive:
        # Run interactive mode
        run_interactive_inference(rollout_policy, config)
    
    print("\nInference complete!")


if __name__ == "__main__":
    main()