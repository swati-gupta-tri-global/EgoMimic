#! /usr/bin/env python3
"""
EgoMimic Policy Server for LBM Evaluation

This policy server wraps a trained EgoMimic model and exposes it through the
LBM gRPC policy interface for robot evaluation.

Usage:
    python grpc_workspace/egomimic_policy_server.py \
        --ckpt_path /path/to/model.ckpt \
        --data_type 0 \
        --camera_name scene_right_0 
    python3 -m grpc_workspace.egomimic_policy_server --ckpt_path ../../trained_models_highlevel/test/None_DT_2026-01-15-19-13-11/models/model_epoch_epoch=169.ckpt --data_type 0 --camera_name scene_right_0 --save_viz --debug
"""
import argparse
import copy
import uuid
import warnings
import os
import sys

import numpy as np
import torch
import cv2

# Try to import imageio for MP4 video creation
try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False
    print("[Warning] imageio not available, videos will be saved as AVI")

# Add EgoMimic and robomimic to path
egomimic_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
robomimic_root = os.path.join(egomimic_root, "external/robomimic")
sys.path.insert(0, egomimic_root)
sys.path.insert(0, robomimic_root)

from grpc_workspace.lbm_policy_server import (
    run_policy_server,
    LbmPolicyServerConfig,
)
from robot_gym.multiarm_spaces import MultiarmObservation, PosesAndGrippers
from robot_gym.policy import Policy, PolicyMetadata

# Import EgoMimic modules
from egomimic_inference import load_model_for_rollout, EgoMimicRolloutPolicy
import robomimic.utils.obs_utils as ObsUtils

def transform_world_to_camera(point_world, T_world_to_cam):
    """
    Transform 3D point from world frame to camera frame.
    
    Args:
        point_world: 3D point (x, y, z) in world frame
        T_world_to_cam: 4x4 transformation matrix from world to camera
    
    Returns:
        point_cam: 3D point in camera frame
    """
    # Convert to homogeneous coordinates
    point_hom = np.append(point_world, 1.0)
    # Transform to camera frame
    point_cam_hom = T_world_to_cam @ point_hom
    return point_cam_hom[:3]

def project_3d_to_2d(point_3d, K, h, w):
    """
    Project 3D point in camera frame to 2D image coordinates.
    
    Args:
        point_3d: 3D point (x, y, z) in camera frame
        K: 3x3 camera intrinsics matrix
    
    Returns:
        (u, v): 2D pixel coordinates, or None if behind camera
    """
    x, y, z = point_3d
    if z <= 0:  # Point behind camera
        return None
    
    # Project using intrinsics: [u, v, 1]^T = K * [x/z, y/z, 1]^T
    u = int(K[0, 0] * x / z + K[0, 2])
    v = int(K[1, 1] * y / z + K[1, 2])
    
    # Check if within image bounds
    if 0 <= u < w and 0 <= v < h:
        return (u, v)
    return None

def _get_policy_metadata(ckpt_path):
    """Create policy metadata from checkpoint path."""
    return PolicyMetadata(
        name="EgoMimic",
        skill_type="BimanualManipulation",
        checkpoint_path=ckpt_path,
        git_repo="EgoMimic",
        git_sha="main",
    )


class EgoMimicPolicy(Policy):
    """Policy wrapper for EgoMimic model inference."""

    def __init__(self, ckpt_path, data_type=0, camera_name="scene_right_0", debug=False, save_viz=False, viz_dir=None):
        """
        Initialize EgoMimic policy.
        
        Args:
            ckpt_path: Path to model checkpoint
            data_type: 0 for robot data, 1 for hand data
            camera_name: Name of camera observation to use from MultiarmObservation
            debug: Enable debug output
            save_viz: Save visualization of predicted actions
            viz_dir: Directory to save visualizations (default: ./egomimic_viz)
        """
        print(f"[EgoMimicPolicy] Loading model from: {ckpt_path}")
        print(f"[EgoMimicPolicy] Data type: {'robot' if data_type == 0 else 'hand'}")
        print(f"[EgoMimicPolicy] Camera: {camera_name}")
        
        self.ckpt_path = ckpt_path
        self.data_type = data_type
        self.camera_name = camera_name
        self.debug = debug
        self.save_viz = save_viz
        self.viz_dir = viz_dir if viz_dir is not None else "./egomimic_viz"
        
        # Visualization frame collection
        self.viz_frames = []
        self.episode_count = 0
        self._is_shutting_down = False  # Flag to prevent operations during shutdown
        
        # Create viz directory if saving visualizations
        if self.save_viz:
            os.makedirs(self.viz_dir, exist_ok=True)
            print(f"[EgoMimicPolicy] Saving visualizations to: {self.viz_dir}")
        
        # Load model
        self.rollout_policy, self.config, self.shape_meta = load_model_for_rollout(
            ckpt_path, data_type=data_type
        )
        
        # Set device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[EgoMimicPolicy] Using device: {self.device}")
        
        # Move model to device
        if hasattr(self.rollout_policy.policy, 'to'):
            self.rollout_policy.policy.to(self.device)
        
        # Initialize state
        self.reset()
        
        print(f"[EgoMimicPolicy] Initialized successfully")

    def reset(self):
        """Reset policy state."""
        # Save video from previous episode if we have frames
        if self.save_viz and len(self.viz_frames) > 0:
            self._save_episode_video()
        
        # Reset counters and buffers
        self._counter = 0
        self._initial_poses = None
        self.viz_frames = []
        self.rollout_policy.reset()
        
        if self.debug:
            print(f"[EgoMimicPolicy] reset() called")

    def get_policy_metadata(self):
        """Return policy metadata."""
        return _get_policy_metadata(self.ckpt_path)
    
    def _save_episode_video(self, during_shutdown=False):
        """Save collected visualization frames as a video."""
        if len(self.viz_frames) == 0:
            return
        
        video_filename = f"episode_{self.episode_count:04d}.mp4"
        video_path = os.path.join(self.viz_dir, video_filename)
        
        h, w = self.viz_frames[0].shape[:2]
        fps = 10
        
        # During shutdown, skip imageio as it doesn't work reliably
        # Try imageio first for MP4 (uses bundled ffmpeg) - only if not during shutdown
        if HAS_IMAGEIO and not during_shutdown:
            try:
                with imageio.get_writer(video_path, fps=fps, codec='libx264', quality=8) as writer:
                    for frame in self.viz_frames:
                        writer.append_data(frame)
                
                if os.path.exists(video_path) and os.path.getsize(video_path) > 0:
                    file_size = os.path.getsize(video_path)
                    print(f"[EgoMimicPolicy] Video saved: {video_path} ({file_size / 1024:.2f} KB)")
                    self.episode_count += 1
                    return
            except Exception as e:
                # Don't print full error during shutdown to avoid noise
                if not during_shutdown:
                    print(f"[Warning] imageio failed: {e}, Exiting")
    
        self.episode_count += 1
    
    def __del__(self):
        """Cleanup: save any remaining frames when policy is destroyed."""
        if hasattr(self, 'save_viz') and self.save_viz and hasattr(self, 'viz_frames') and len(self.viz_frames) > 0:
            print("[EgoMimicPolicy] Saving final episode video on cleanup")
            # Use OpenCV directly during shutdown since imageio doesn't work reliably
            self._save_episode_video(during_shutdown=True)

    def _extract_image_from_observation(self, observation: MultiarmObservation) -> np.ndarray:
        """
        Extract RGB image from MultiarmObservation.
        
        Args:
            observation: MultiarmObservation from robot
            
        Returns:
            image: RGB image as numpy array (H, W, 3)
        """
        # Get camera observation
        if self.camera_name not in observation.visuo:
            available_cameras = list(observation.visuo.keys())
            raise ValueError(
                f"Camera '{self.camera_name}' not found in observation. "
                f"Available cameras: {available_cameras}"
            )
        
        camera_obs = observation.visuo[self.camera_name]
        image = camera_obs.rgb.array  # Should be (H, W, 3) uint8
        
        if self.debug and self._counter == 0:
            print(f"[EgoMimicPolicy] Image shape: {image.shape}, dtype: {image.dtype}")
        
        return image

    def _prepare_observation(self, image: np.ndarray, observation: MultiarmObservation) -> dict:
        """
        Prepare observation dictionary for EgoMimic model.
        
        Args:
            image: RGB image (H, W, 3)
            observation: Full robot observation (for extracting proprioception)
            
        Returns:
            obs_dict: Formatted observation dictionary
        """
        # Resize image to model's expected size if needed
        # EgoMimic typically uses 640x480 or similar
        target_size = (640, 480)  # (W, H)
        if image.shape[:2][::-1] != target_size:
            image = cv2.resize(image, target_size)
        
        # Convert to float and normalize
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        
        # Convert HWC to CHW
        image = np.transpose(image, (2, 0, 1))
        
        # Convert to tensor
        image_tensor = torch.from_numpy(image).float().unsqueeze(0)  # Add batch dimension
        image_tensor = image_tensor.to(self.device)
        
        # Create observation dictionary
        obs_dict = {}
        
        # Add RGB observation - use the config's expected key
        rgb_keys = self.config.observation.modalities.obs.rgb
        for rgb_key in rgb_keys:
            obs_dict[rgb_key] = image_tensor
        
        # Add low-dim observations - extract real values from robot observation
        low_dim_keys = self.config.observation.modalities.obs.low_dim
        for low_dim_key in low_dim_keys:
            if low_dim_key == "joint_positions":
                # Extract actual joint positions from robot observation
                # Assuming observation.robot.actual has joint information
                if hasattr(observation.robot.actual, 'joint_position'):
                    joint_positions = observation.robot.actual.joint_position
                    # append the arrays for each key 

                    concat_joint_positions = np.concatenate((joint_positions['left::panda'], joint_positions['right::panda']))
                    # ipdb> joint_positions
                    # {'right::panda': array([ 0.43633231, -0.61086524,  0.        , -2.35619449,  0.        ,
                    #         1.83259571, -1.13446401]), 'left::panda': array([-0.61086524, -0.61086524,  0.        , -2.35619449,  0.        ,
                    #         1.83259571,  0.78539816])}
                    print (f"[EgoMimicPolicy] joint_position keys in obs: {joint_positions.keys()}")
                    # Convert to tensor - expecting 14 joints (7 per arm)
                    joint_tensor = torch.from_numpy(concat_joint_positions).float().unsqueeze(0).to(self.device)
                else:
                    # Fallback to zeros if not available
                    if self.debug and self._counter == 0:
                        print("[Warning] joint_positions not found in observation, using zeros")
                    joint_tensor = torch.zeros(1, 14).float().to(self.device)
                obs_dict[low_dim_key] = joint_tensor
                
            elif low_dim_key == "ee_pose":
                # Extract actual end-effector poses from robot observation
                # Get poses for both arms and flatten to 6D (left_xyz, right_xyz)
                robot_names = list(observation.robot.actual.poses.keys())
                if len(robot_names) >= 2:
                    left_pos = observation.robot.actual.poses[robot_names[0]].translation()
                    right_pos = observation.robot.actual.poses[robot_names[1]].translation()
                    ee_pose = np.concatenate([left_pos, right_pos])  # (6,)
                    ee_tensor = torch.from_numpy(ee_pose).float().unsqueeze(0).to(self.device)
                else:
                    if self.debug and self._counter == 0:
                        print("[Warning] Could not extract ee_pose from observation, using zeros")
                    ee_tensor = torch.zeros(1, 6).float().to(self.device)
                obs_dict[low_dim_key] = ee_tensor
        
        return obs_dict

    def _extract_action_data(self, action_dict: dict, observation: MultiarmObservation) -> dict:
        """
        Extract and parse action data from model predictions.
        
        Args:
            action_dict: Dictionary with action predictions
            observation: Current observation (for robot/gripper names)
            
        Returns:
            Dictionary with parsed action components:
                - 'left_xyz_cam': Left arm target position in camera frame (3,)
                - 'right_xyz_cam': Right arm target position in camera frame (3,)
                - 'left_pos_world': Left arm current position in world frame (3,)
                - 'right_pos_world': Right arm current position in world frame (3,)
                - 'gripper_left': Left gripper value
                - 'gripper_right': Right gripper value
                - 'robot_names': List of robot names [left, right]
                - 'gripper_names': List of gripper names [left, right]
        """
        result = {}
        timestep_taken = -1
        horizon_secs = 4
        sim_controller_freq = 10 # 10 Hz
        # Get robot names
        robot_names = list(observation.robot.actual.poses.keys())
        result['robot_names'] = robot_names
        
        # Get current positions in world frame
        if len(robot_names) >= 2:
            result['left_pos_world'] = observation.robot.actual.poses[robot_names[0]].translation()
            result['right_pos_world'] = observation.robot.actual.poses[robot_names[1]].translation()
        else:
            result['left_pos_world'] = np.zeros(3)
            result['right_pos_world'] = np.zeros(3)
        
        # Parse XYZ actions (absolute positions in camera frame)
        if 'actions_xyz_act' in action_dict:
            xyz_actions = action_dict['actions_xyz_act']
            
            # Take last timestep if sequence
            if len(xyz_actions.shape) == 2:
                seq_len = xyz_actions.shape[0]
                timestep_taken = seq_len // (horizon_secs * sim_controller_freq)
                xyz_actions = xyz_actions[timestep_taken]  # (6,) - [left_xyz(3), right_xyz(3)]
            
            result['left_xyz_cam'] = xyz_actions[0:3]
            result['right_xyz_cam'] = xyz_actions[3:6]
        else:
            result['left_xyz_cam'] = None
            result['right_xyz_cam'] = None
        
        # Parse gripper actions
        result['gripper_left'] = None
        result['gripper_right'] = None
        result['gripper_names'] = list(observation.robot.actual.grippers.keys())
        
        if 'actions_joints_act' in action_dict:
            joint_actions = action_dict['actions_joints_act']
            
            # Take last timestep if sequence
            if len(joint_actions.shape) == 2:
                seq_len = joint_actions.shape[0]
                timestep_taken = seq_len // (horizon_secs * sim_controller_freq)  
                joint_actions = joint_actions[timestep_taken]  # (16,)
            
            # Extract gripper values at indices 7 and 15
            result['gripper_left'] = joint_actions[7]
            result['gripper_right'] = joint_actions[15]

            # import ipdb; ipdb.set_trace()
            print (f"[EgoMimicPolicy] Extracted gripper actions: left={result['gripper_left']}, right={result['gripper_right']}")
        
        return result

    def _parse_action_to_poses(
        self, 
        action_data: dict, 
        initial_poses: dict,
        observation: MultiarmObservation,
        extrinsics=None
    ) -> PosesAndGrippers:
        """
        Convert parsed action data to robot poses.
        
        Args:
            action_data: Dictionary from _extract_action_data()
            initial_poses: Initial robot poses
            observation: Current observation
            
        Returns:
            PosesAndGrippers with commanded poses and grippers
        """
        poses = copy.deepcopy(initial_poses)
        grippers = copy.deepcopy(observation.robot.actual.grippers)
        
        # Set XYZ positions if available
        if action_data['left_xyz_cam'] is not None and action_data['right_xyz_cam'] is not None:
            robot_names = action_data['robot_names']
            if len(robot_names) >= 2:
                left_robot = robot_names[0]
                right_robot = robot_names[1]
                
                # EgoMimic predicts absolute XYZ positions in camera frame
                if extrinsics is not None:      
                    # apply inverse camera extrinsics transformation to convert to world frame 
                    left_xyz_world = transform_world_to_camera(action_data['left_xyz_cam'], extrinsics)
                    right_xyz_world = transform_world_to_camera(action_data['right_xyz_cam'], extrinsics)
                    
                    poses[left_robot].set_translation(left_xyz_world)
                    poses[right_robot].set_translation(right_xyz_world)
                    
                    if self.debug:
                        print(f"[EgoMimicPolicy] {left_robot} current: {action_data['left_pos_world']}, target (world frame): {left_xyz_world}")
                        print(f"[EgoMimicPolicy] {right_robot} current: {action_data['right_pos_world']}, target (world frame): {right_xyz_world}")
                        delta_left = np.linalg.norm(left_xyz_world - action_data['left_pos_world'])
                        delta_right = np.linalg.norm(right_xyz_world - action_data['right_pos_world'])
                        print(f"[EgoMimicPolicy] Delta magnitudes - left: {delta_left:.4f}, right: {delta_right:.4f}")
                else:
                    # Directly set positions in camera frame (may not be correct)
                    poses[left_robot].set_translation(action_data['left_xyz_cam'])
                    poses[right_robot].set_translation(action_data['right_xyz_cam'])
                    if self.debug:
                        print(f"[EgoMimicPolicy] {left_robot} current: {action_data['left_pos_world']}, target (cam frame): {action_data['left_xyz_cam']}")
                        print(f"[EgoMimicPolicy] {right_robot} current: {action_data['right_pos_world']}, target (cam frame): {action_data['right_xyz_cam']}")
                        delta_left = np.linalg.norm(action_data['left_xyz_cam'] - action_data['left_pos_world'])
                        delta_right = np.linalg.norm(action_data['right_xyz_cam'] - action_data['right_pos_world'])
                        print(f"[EgoMimicPolicy] Delta magnitudes - left: {delta_left:.4f}, right: {delta_right:.4f}")
        
        # Set gripper states if available (0.05 works better)
        gripper_threshold = 0.5  # Threshold to determine open/close
        if action_data['gripper_left'] is not None and action_data['gripper_right'] is not None:
            gripper_names = action_data['gripper_names']
            if len(gripper_names) >= 2:
                left_gripper_name = gripper_names[0]
                right_gripper_name = gripper_names[1]
                
                # Map to [0, 1] range where 0=closed, 1=open
                grippers[left_gripper_name] = 1.0 if float(action_data['gripper_left']) > gripper_threshold else 0.0
                grippers[right_gripper_name] = 1.0 if float(action_data['gripper_right']) > gripper_threshold else 0.0
                
                if self.debug and self._counter % 10 == 0:
                    print(f"[EgoMimicPolicy] Gripper actions - {left_gripper_name}: {action_data['gripper_left']}, {right_gripper_name}: {action_data['gripper_right']}")
        
        return PosesAndGrippers(poses=poses, grippers=grippers)
    
    def _visualize_actions(self, image: np.ndarray, action_data: dict, observation: MultiarmObservation, intrinsics=None, extrinsics=None) -> np.ndarray:
        """
        Visualize predicted actions on the image and save.
        Projects 3D positions to image space using camera intrinsics.
        
        Args:
            image: RGB image (H, W, 3) uint8
            action_data: Dictionary from _extract_action_data()
            observation: Current observation
        """
        viz_image = image.copy()
        
        # Skip visualization if no XYZ actions
        if action_data['left_xyz_cam'] is None or action_data['right_xyz_cam'] is None:
            return viz_image
        
        # Transform current positions from world frame to camera frame if extrinsics available
        if extrinsics is not None:
            left_pos_current_cam = transform_world_to_camera(action_data['left_pos_world'], np.linalg.inv(extrinsics))
            right_pos_current_cam = transform_world_to_camera(action_data['right_pos_world'], np.linalg.inv(extrinsics))
        else:
            print ("[Warning] No extrinsics provided, assuming current positions are in camera frame")
            # If no extrinsics, assume current positions are already in camera frame
            # (This is likely incorrect but allows visualization to proceed)
            left_pos_current_cam = action_data['left_pos_world']
            right_pos_current_cam = action_data['right_pos_world']
        
        h, w = viz_image.shape[:2]
        # Project current positions in camera frame to 2D (green for left, blue for right)
        left_current_2d = project_3d_to_2d(left_pos_current_cam, intrinsics, h, w)
        right_current_2d = project_3d_to_2d(right_pos_current_cam, intrinsics, h, w)
        
        # Project target positions (predicted actions are already in camera frame)
        pred_left_2d = project_3d_to_2d(action_data['left_xyz_cam'], intrinsics, h, w)
        pred_right_2d = project_3d_to_2d(action_data['right_xyz_cam'], intrinsics, h, w)
        
        # Draw current positions
        if left_current_2d is not None:
            cv2.circle(viz_image, left_current_2d, 12, (0, 255, 0), -1)  # Green filled circle
            cv2.circle(viz_image, left_current_2d, 14, (255, 255, 255), 2)  # White outline
            cv2.putText(viz_image, "Lc", (left_current_2d[0] - 8, left_current_2d[1] + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        if right_current_2d is not None:
            cv2.circle(viz_image, right_current_2d, 12, (255, 0, 0), -1)  # Red filled circle
            cv2.circle(viz_image, right_current_2d, 14, (255, 255, 255), 2)  # White outline
            cv2.putText(viz_image, "Rc", (right_current_2d[0] - 8, right_current_2d[1] + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Draw target positions
        if pred_left_2d is not None:
            cv2.circle(viz_image, pred_left_2d, 10, (0, 255, 255), -1)  # Cyan filled circle
            cv2.circle(viz_image, pred_left_2d, 12, (255, 255, 255), 2)  # White outline
            cv2.putText(viz_image, "Lap", (pred_left_2d[0] - 10, pred_left_2d[1] - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            # Draw arrow from current to target
            if left_current_2d is not None:
                cv2.arrowedLine(viz_image, left_current_2d, pred_left_2d, 
                               (0, 255, 0), 2, tipLength=0.3)
        
        if pred_right_2d is not None:
            cv2.circle(viz_image, pred_right_2d, 10, (255, 255, 0), -1)  # Yellow filled circle
            cv2.circle(viz_image, pred_right_2d, 12, (255, 255, 255), 2)  # White outline
            cv2.putText(viz_image, "Rap", (pred_right_2d[0] - 10, pred_right_2d[1] - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            
            # Draw arrow from current to target
            if right_current_2d is not None:
                cv2.arrowedLine(viz_image, right_current_2d, pred_right_2d,
                               (255, 0, 0), 2, tipLength=0.3)
        
        # Draw gripper states if available
        if action_data['gripper_left'] is not None and action_data['gripper_right'] is not None:
            cv2.putText(viz_image, f"Gripper L: {action_data['gripper_left']:.3f}  R: {action_data['gripper_right']:.3f}",
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add legend
        h, w = viz_image.shape[:2]
        legend_x = w - 200
        cv2.putText(viz_image, "Current pose: Lc(green) Rc(Red)", (legend_x, h - 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(viz_image, "Pred actions: Lap(cyan) Rap(yellow)", (legend_x, h - 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Add frame counter
        cv2.putText(viz_image, f"Frame: {self._counter}",
                   (w - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
        return viz_image


    def step(self, observation: MultiarmObservation) -> PosesAndGrippers:
        """
        Execute one policy step.
        
        Args:
            observation: Current observation from robot
            
        Returns:
            PosesAndGrippers: Commanded poses and gripper states
        """
        if self.debug and self._counter % 10 == 0:
            print(f"[EgoMimicPolicy step] timestep: {self._counter}")
        
        # Store initial poses on first step
        if self._initial_poses is None:
            self._initial_poses = copy.deepcopy(observation.robot.actual.poses)
        
        # Extract image from observation
        image = self._extract_image_from_observation(observation)
        
        # Prepare observation for model (pass full observation to extract proprioception)
        obs_dict = self._prepare_observation(image, observation)
        
        # Get action prediction from model
        action_output = self.rollout_policy.get_action(obs_dict)
        
        # Parse action output
        if isinstance(action_output, dict):
            # Already parsed (contains 'actions_xyz_act', 'actions_joints_act', etc.)
            pred_action_dict = action_output
        else:
            # Raw numpy array - parse it
            from egomimic_inference import parse_egomimic_actions
            pred_action_dict = parse_egomimic_actions(action_output)
        
        # Extract action data (gets the last timestep if sequence)
        action_data = self._extract_action_data(pred_action_dict, observation)

        camera_obs = observation.visuo[self.camera_name]
        # import ipdb; ipdb.set_trace()
        # ipdb> camera_obs.rgb.K
        #     array([[616.12902832,   0.        , 321.26901245],
        #         [  0.        , 615.75799561, 247.86399841],
        #         [  0.        ,   0.        ,   1.        ]])
        # ipdb> camera_obs.rgb.X_TC
        #     RigidTransform(
        #     R=RotationMatrix([
        #         [0.020032611070631746, -0.6970191518623627, 0.716772625335793],
        #         [-0.9996948973692944, -0.0036028314345558045, 0.024436279983867368],
        #         [-0.014450144203037632, -0.7170434586151152, -0.696878663606359],
        #     ]),
        #     p=[-0.8282514233472894, -0.015489063618070866, 0.8799827656809344],
        #     )
        intrinsics = camera_obs.rgb.K
        extrinsics_rigid_transform = camera_obs.rgb.X_TC
        R = extrinsics_rigid_transform.rotation().matrix()  # 3x3 rotation
        p = extrinsics_rigid_transform.translation()        # 3D translation
        extrinsics = np.eye(4)
        extrinsics[:3, :3] = R
        extrinsics[:3, 3] = p
        
        # Visualize actions if enabled
        viz_image = None
        if self.save_viz:
            viz_image = self._visualize_actions(image, action_data, observation, intrinsics, extrinsics)
            # write visualization images via cv2
            cv2.imwrite(os.path.join(self.viz_dir, f"frame_{self._counter:05d}.jpg"), cv2.cvtColor(viz_image, cv2.COLOR_RGB2BGR))

            # Collect frame for video
            self.viz_frames.append(viz_image.copy())
        
        # Convert action data to poses
        result = self._parse_action_to_poses(action_data, self._initial_poses, observation, extrinsics)
        
        self._counter += 1
        return result


class EgoMimicPolicyBatch(Policy):
    """
    Batch version of EgoMimic policy for gRPC interface.
    Supports multiple parallel evaluation environments.
    """

    def __init__(self, ckpt_path, data_type=0, camera_name="scene_right_0", debug=False, save_viz=False, viz_dir=None):
        """
        Initialize batch EgoMimic policy.
        
        Args:
            ckpt_path: Path to model checkpoint
            data_type: 0 for robot data, 1 for hand data
            camera_name: Name of camera observation to use
            debug: Enable debug output
            save_viz: Save visualization of predicted actions
            viz_dir: Directory to save visualizations
        """
        self.ckpt_path = ckpt_path
        self.data_type = data_type
        self.camera_name = camera_name
        self.debug = debug
        self.save_viz = save_viz
        self.viz_dir = viz_dir
        
        # Internal client identifier for non-batch interface
        self._internal_uuid = uuid.uuid4()
        
        # Mapping from UUID to individual policy per UUID
        self._sub_policies: dict[uuid.UUID, EgoMimicPolicy] = {}

    def reset(self, seed: int | None = None, options=None):
        """Reset single environment (non-batch interface)."""
        self.reset_batch({self._internal_uuid: seed}, options)

    def reset_batch(
        self, 
        seeds: dict[uuid.UUID, int | None], 
        options=None
    ) -> None:
        """
        Reset multiple environments.
        
        Args:
            seeds: Dictionary mapping UUID to random seed
            options: Optional reset parameters
        """
        for one_uuid, one_seed in seeds.items():
            if one_seed is not None:
                warnings.warn(f"Random seed ignored for {one_uuid}")

            # Create or reset policy for this UUID
            if one_uuid not in self._sub_policies:
                # Create unique viz dir for each UUID if saving visualizations
                uuid_viz_dir = None
                if self.save_viz:
                    uuid_viz_dir = os.path.join(self.viz_dir, str(one_uuid)[:8])
                
                self._sub_policies[one_uuid] = EgoMimicPolicy(
                    ckpt_path=self.ckpt_path,
                    data_type=self.data_type,
                    camera_name=self.camera_name,
                    debug=self.debug,
                    save_viz=self.save_viz,
                    viz_dir=uuid_viz_dir
                )
            else:
                self._sub_policies[one_uuid].reset()

    def get_policy_metadata(self):
        """Return policy metadata."""
        return _get_policy_metadata(self.ckpt_path)

    def step(self, observation: MultiarmObservation) -> PosesAndGrippers:
        """Execute single environment step (non-batch interface)."""
        batch_actions = self.step_batch({self._internal_uuid: observation})
        return batch_actions[self._internal_uuid]

    def step_batch(
        self, 
        observations: dict[uuid.UUID, MultiarmObservation]
    ) -> dict[uuid.UUID, PosesAndGrippers]:
        """
        Execute multiple environment steps in batch.
        
        Args:
            observations: Dictionary mapping UUID to observation
            
        Returns:
            Dictionary mapping UUID to actions
        """
        batch_actions = {}
        for one_uuid, observation in observations.items():
            sub_policy = self._sub_policies[one_uuid]
            batch_actions[one_uuid] = sub_policy.step(observation)
        
        return batch_actions


def main():
    parser = argparse.ArgumentParser(
        description="EgoMimic Policy Server for LBM Evaluation"
    )
    
    # EgoMimic-specific arguments
    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="Path to EgoMimic model checkpoint (.ckpt file)"
    )
    parser.add_argument(
        "--data_type",
        type=int,
        default=0,
        choices=[0, 1],
        help="Data type: 0=robot, 1=hand (default: 0)"
    )
    parser.add_argument(
        "--camera_name",
        type=str,
        default="scene_right_0",
        help="Camera name to use from observation (default: scene_right_0)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output"
    )
    parser.add_argument(
        "--save_viz",
        action="store_true",
        help="Save visualization of predicted actions overlaid on images"
    )
    parser.add_argument(
        "--viz_dir",
        type=str,
        default="./egomimic_viz",
        help="Directory to save visualizations (default: ./egomimic_viz)"
    )
    
    # Add LBM policy server arguments
    LbmPolicyServerConfig.add_argparse_arguments(parser)
    
    args = parser.parse_args()
    
    # Validate checkpoint exists
    if not os.path.exists(args.ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt_path}")
    
    print("=" * 80)
    print("EgoMimic Policy Server")
    print("=" * 80)
    print(f"Checkpoint: {args.ckpt_path}")
    print(f"Data type: {'robot' if args.data_type == 0 else 'hand'}")
    print(f"Camera: {args.camera_name}")
    print(f"Debug: {args.debug}")
    print(f"Save visualizations: {args.save_viz}")
    if args.save_viz:
        print(f"Visualization directory: {args.viz_dir}")
    print("=" * 80)
    
    # Create batch policy
    policy = EgoMimicPolicyBatch(
        ckpt_path=args.ckpt_path,
        data_type=args.data_type,
        camera_name=args.camera_name,
        debug=args.debug,
        save_viz=args.save_viz,
        viz_dir=args.viz_dir
    )
    
    # Run policy server
    run_policy_server(policy, args)


if __name__ == "__main__":
    main()
