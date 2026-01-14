#!/usr/bin/env python3
"""
Visualization script with minimal dependencies to load a single LBM episode in spartan format and visualize the arm's
end-effector poses + action trajectories projected onto RGB images, 
using camera intrinsics and extrinsics and generate a video

Usage:
Get data from:
# AWS_REGION=us-east-1 aws s3 cp s3://robotics-manip-lbm/efs/data/tasks/BimanualHangMugsOnMugHolderFromDryingRack/riverway/sim/bc/teleop/2024-12-16T11-49-42-05-00/diffusion_spartan/episode_0/processed/ . --recursive

Then run:
docker exec -it swati-egomimic python /workspace/externals/EgoMimic/visualize_lbm_episode.py \
    --episode_path data/tasks/BimanualHangMugsOnMugHolderFromDryingRack/riverway/sim/bc/teleop/2024-12-16T11-49-42-05-00/diffusion_spartan/episode_0/processed/ \
    --output_dir /workspace/externals/EgoMimic/validation_videos \
    --max_frames 200
"""

import os
import numpy as np
import argparse
import yaml
import cv2
import torch
from tqdm import tqdm
import torchvision
import matplotlib.pyplot as plt
import av

def write_video_safe(filename, video_array, fps=30):
    """
    Write video with PyAV compatibility handling for different versions.
    
    Args:
        filename: Output video file path
        video_array: Tensor of shape (T, H, W, 3) with values in [0, 255]
        fps: Frames per second
    """
    try:
        # Try using torchvision's write_video directly
        torchvision.io.write_video(filename, video_array, fps=fps)
    except TypeError as e:
        if "an integer is required" in str(e):
            # PyAV 16+ requires manual video writing due to pict_type API change
            print(f"Warning: Using fallback video writer due to PyAV compatibility issue")
            
            # Convert to numpy and ensure uint8
            if isinstance(video_array, torch.Tensor):
                video_array = video_array.cpu().numpy()
            video_array = video_array.astype(np.uint8)
            
            # Open output container
            container = av.open(filename, mode='w')
            stream = container.add_stream('h264', rate=fps)
            stream.height = video_array.shape[1]
            stream.width = video_array.shape[2]
            stream.pix_fmt = 'yuv420p'
            
            # Write frames
            for frame_array in video_array:
                frame = av.VideoFrame.from_ndarray(frame_array, format='rgb24')
                for packet in stream.encode(frame):
                    container.mux(packet)
            
            # Flush stream
            for packet in stream.encode():
                container.mux(packet)
            
            container.close()
        else:
            raise

def interpolate_arr(v, seq_length):
    """
    v: (B, T, D)
    seq_length: int
    """
    assert len(v.shape) == 3
    if v.shape[1] == seq_length:
        return v
    
    interpolated = []
    for i in range(v.shape[0]):
        index = v[i]

        interp = scipy.interpolate.interp1d(
            np.linspace(0, 1, index.shape[0]), index, axis=0
        )
        interpolated.append(interp(np.linspace(0, 1, seq_length)))

    # size (B, seq_length, D)
    return np.array(interpolated)

def ee_pose_to_cam_frame(ee_pose_base, T_cam_base):
    """
    ee_pose_base: (N, 3)
    T_cam_base: (4, 4)

    returns ee_pose_cam: (N, 3)
    """
    N, _ = ee_pose_base.shape
    ee_pose_base = np.concatenate([ee_pose_base, np.ones((N, 1))], axis=1)

    if T_cam_base.ndim == 2:  # Single transform (4, 4)
        ee_pose_grip_cam = np.linalg.inv(T_cam_base) @ ee_pose_base.T
        return ee_pose_grip_cam.T[:, :3]
    else:
        T_cam_inv = np.linalg.inv(T_cam_base)  # (N, 4, 4)
        # Batch matrix multiplication: (N, 4, 4) @ (N, 4, 1)
        ee_pose_grip_cam = np.einsum('nij,nj->ni', T_cam_inv, ee_pose_base)
        return ee_pose_grip_cam[:, :3]

def cam_frame_to_cam_pixels(ee_pose_cam, intrinsics):
    """
    camera frame 3d coordinates to pixels in camera frame
    ee_pose_cam: (N, 3)
    intrinsics: 3x4 matrix
    """
    # Handle case where intrinsics has extra batch dimension
    if intrinsics.ndim == 3 and intrinsics.shape[0] == 1:
        intrinsics = intrinsics.squeeze(0)  # Remove batch dimension
    
    N, _ = ee_pose_cam.shape
    ee_pose_cam = np.concatenate([ee_pose_cam, np.ones((N, 1))], axis=1)

    px_val = intrinsics @ ee_pose_cam.T
    
    # Safety check: avoid division by very small Z values that cause overflow
    z_vals = px_val[2, :]
    eps = 1e-6  # Small epsilon to prevent division by near-zero values
    safe_z = np.where(np.abs(z_vals) < eps, eps * np.sign(z_vals), z_vals)
    safe_z = np.where(safe_z == 0, eps, safe_z)  # Handle exact zeros
    
    px_val = px_val / safe_z

    return px_val.T


def draw_dot_on_frame(frame, pixel_vals, show=True, palette="Purples", dot_size=5):
    """
    frame: (H, W, C) numpy array
    pixel_vals: (N, 2) numpy array of pixel values to draw on frame
    Drawn in light to dark order
    """
    frame = frame.astype(np.uint8).copy()
    if isinstance(pixel_vals, tuple):
        pixel_vals = [pixel_vals]

    # get purples color palette, and color the circles accordingly
    color_palette = plt.get_cmap(palette)
    color_palette = color_palette(np.linspace(0, 1, len(pixel_vals)))
    color_palette = (color_palette[:, :3] * 255).astype(np.uint8)
    color_palette = color_palette.tolist()

    dots_drawn = 0
    dots_skipped_extreme = 0
    dots_skipped_bounds = 0

    for i, pixel_val in enumerate(pixel_vals):
        try:
            # Check if pixel coordinates are within reasonable bounds
            x, y = int(pixel_val[0]), int(pixel_val[1])
            
            # Skip drawing if coordinates are extreme (likely overflow)
            if abs(x) > 1e6 or abs(y) > 1e6:
                dots_skipped_extreme += 1
                continue
                
            # Skip if coordinates are outside reasonable image bounds
            if x < -1000 or x > frame.shape[1] + 1000 or y < -1000 or y > frame.shape[0] + 1000:
                dots_skipped_bounds += 1
                continue
                
            frame = cv2.circle(
                frame,
                (x, y),
                dot_size,
                color_palette[i],
                -1,
            )
            dots_drawn += 1
        except Exception as e:
            print("Got bad pixel_val: ", pixel_val, "Error:", str(e))
        if show:
            plt.imshow(frame)
            plt.show()

    return frame   

def draw_both_actions_on_frame(im, color, actions, arm="both", intrinsics=None, extrinsics=None, subsample=False):
    if intrinsics is None:
        raise ValueError("Intrinsics must be provided to draw actions on frame")
    
    actions = actions.reshape(-1, 3)
    actions_drawable = cam_frame_to_cam_pixels(actions, intrinsics)
    
    im = draw_dot_on_frame(
        im, actions_drawable, show=False, palette=color
    )

    return im

def load_yaml(file_path):
    with open(file_path, "r") as file:
        data = yaml.safe_load(file)
    return data

def visualize_lbm_episode(episode_path, output_dir, max_frames=None):
    """
    Load a single LBM episode and visualize end-effector poses on RGB images.
    
    Args:
        episode_path: Path to the processed episode directory
        output_dir: Directory to save visualization video
        max_frames: Maximum number of frames to process (None = all frames)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading episode from: {episode_path}")
    
    # Load metadata
    meta_data_file = os.path.join(episode_path, "metadata.yaml")
    if not os.path.isfile(meta_data_file):
        raise FileNotFoundError(f"Metadata file not found: {meta_data_file}")
    
    meta_data = load_yaml(meta_data_file)
    print(f"Metadata loaded: {list(meta_data.keys())}")
    
    # Load observations
    observations_file = os.path.join(episode_path, "observations.npz")
    if not os.path.isfile(observations_file):
        raise FileNotFoundError(f"Observations file not found: {observations_file}")
    
    observations = np.load(observations_file)
    print(f"Observation keys: {list(observations.keys())}")
    
    # Get camera data
    camera_names = {val: key for key, val in 
                   meta_data["camera_id_to_semantic_name"].items()}
    
    camera_name = "scene_right_0"
    if camera_name not in camera_names:
        raise ValueError(f"Camera {camera_name} not found. Available cameras: {list(camera_names.keys())}")
    
    camera_id = camera_names[camera_name]
    print(f"Using camera: {camera_name} (ID: {camera_id})")
    
    # Load camera images and parameters
    front_img = observations[camera_id]  # (N, H, W, 3)
    intrinsics = np.load(os.path.join(episode_path, "intrinsics.npz"))[camera_id]  # (3, 3)
    extrinsics = np.load(os.path.join(episode_path, "extrinsics.npz"))[camera_id][0]  # (4, 4)
    
    print(f"Images shape: {front_img.shape}")
    print(f"Intrinsics shape: {intrinsics.shape}")
    print(f"Extrinsics shape: {extrinsics.shape}")
    
    # Load end-effector poses
    pose_xyz_left = observations["robot__actual__poses__left::panda__xyz"]  # (N, 3)
    pose_xyz_right = observations["robot__actual__poses__right::panda__xyz"]  # (N, 3)

    robot_position_action_left = observations["robot__desired__poses__left::panda__xyz"] # (N, 3)
    robot_position_action_right = observations["robot__desired__poses__right::panda__xyz"]
    
    print(f"Left EE pose shape: {pose_xyz_left.shape}")
    print(f"Right EE pose shape: {pose_xyz_right.shape}")
    print(f"Left EE action shape: {robot_position_action_left.shape}")
    print(f"Right EE action shape: {robot_position_action_right.shape}")

    # Load desired joint positions and actions for chunking
    robot_joint_action_left = observations["robot__desired__joint_position__left::panda"]  # (N, 7)
    robot_joint_action_right = observations["robot__desired__joint_position__right::panda"]  # (N, 7)
    robot_gripper_action_left = observations["robot__desired__grippers__left::panda_hand"]  # (N, 1)
    robot_gripper_action_right = observations["robot__desired__grippers__right::panda_hand"]  # (N, 1)

    # Combine actions for chunking (same format as processing script)
    position_actions = np.hstack([robot_position_action_left, robot_position_action_right])  # (N, 6)
    joint_actions = np.hstack([robot_joint_action_left, robot_joint_action_right])  # (N, 14)
    gripper_actions = np.hstack([robot_gripper_action_left, robot_gripper_action_right])  # (N, 2)
    actions = np.hstack([position_actions, joint_actions, gripper_actions])  # (N, 22)
    
    print(f"Combined actions shape: {actions.shape}")
    
    # Process actions with chunking and interpolation
    horizon_seconds = 4.0
    N_actions = actions.shape[0]
    chunk_size = int(N_actions / horizon_seconds)
    ac_dim = actions.shape[1]
    
    print(f"Chunk size: {chunk_size}, Action dimension: {ac_dim}")
    
    ac_reshape_interp = []
    
    for i in range(0, N_actions):
        if i + chunk_size > N_actions:
            # Not enough data to create another chunk, tile last action
            ac_reshape = np.zeros((1, chunk_size, ac_dim))
            ac_reshape[:, :N_actions - i] = actions[i : N_actions].reshape(1, -1, ac_dim)
            ac_reshape[:, N_actions - i :] = np.tile(
                actions[N_actions - 1].reshape(1, 1, ac_dim), 
                (1, chunk_size - (N_actions - i), 1)
            )
        else:
            ac_reshape = actions[i : i + chunk_size].reshape(1, chunk_size, ac_dim)
        
        # Interpolate to 100 steps
        ac_reshape_interp.append(interpolate_arr(ac_reshape, 100))
    
    ac_reshape_interp = np.concatenate(ac_reshape_interp, axis=0)
    ac_reshape_interp = ac_reshape_interp.astype(np.float32)
    ac_reshape_interp = np.nan_to_num(ac_reshape_interp, nan=0.0, posinf=0.0, neginf=0.0)
    
    print(f"Interpolated actions shape: {ac_reshape_interp.shape}")
    
    # Extract XYZ actions from interpolated chunks
    left_xyz_act = ac_reshape_interp[:, :, :3]  # (N, 100, 3)
    right_xyz_act = ac_reshape_interp[:, :, 3:6]  # (N, 100, 3)
    
    # Transform action chunks to camera frame
    # Reshape to (N*100, 3) for batch transformation
    left_xyz_act_flat = left_xyz_act.reshape(-1, 3)  # (N*100, 3)
    right_xyz_act_flat = right_xyz_act.reshape(-1, 3)  # (N*100, 3)
    
    left_xyz_act_cam_flat = ee_pose_to_cam_frame(left_xyz_act_flat, extrinsics)[:, :3]  # (N*100, 3)
    right_xyz_act_cam_flat = ee_pose_to_cam_frame(right_xyz_act_flat, extrinsics)[:, :3]  # (N*100, 3)
    
    # Reshape back to (N, 100, 3)
    left_xyz_act_cam = left_xyz_act_cam_flat.reshape(N_actions, 100, 3)  # (N, 100, 3)
    right_xyz_act_cam = right_xyz_act_cam_flat.reshape(N_actions, 100, 3)  # (N, 100, 3)
    
    combined_xyz_act_cam = np.concatenate([left_xyz_act_cam, right_xyz_act_cam], axis=2)  # (N, 100, 6)
    
    print(f"Combined XYZ action trajectory shape (camera frame): {combined_xyz_act_cam.shape}")

    # Transform poses to camera frame
    pose_xyz_left_cam = ee_pose_to_cam_frame(pose_xyz_left, extrinsics)[:, :3]  # (N, 3)
    pose_xyz_right_cam = ee_pose_to_cam_frame(pose_xyz_right, extrinsics)[:, :3]  # (N, 3)
    
    # Transform single-step actions to camera frame (for comparison)
    action_xyz_left_cam = ee_pose_to_cam_frame(robot_position_action_left, extrinsics)[:, :3]  # (N, 3)
    action_xyz_right_cam = ee_pose_to_cam_frame(robot_position_action_right, extrinsics)[:, :3]  # (N, 3)
    
    # Combine left and right EE poses
    ee_pose = np.hstack([pose_xyz_left_cam, pose_xyz_right_cam])  # (N, 6)
    
    # Combine left and right single-step actions
    ee_actions = np.hstack([action_xyz_left_cam, action_xyz_right_cam])  # (N, 6)
    
    print(f"Combined EE pose in camera frame shape: {ee_pose.shape}")
    print(f"Combined EE actions in camera frame shape: {ee_actions.shape}")
    
    # Determine number of frames to process
    N = front_img.shape[0]
    if max_frames is not None:
        N = min(N, max_frames)
    
    print(f"Processing {N} frames...")
    
    # Convert 3x3 intrinsics to 3x4 projection matrix if needed
    intrinsics_proj = intrinsics
    if intrinsics_proj.shape == (3, 3):
        intrinsics_proj = np.hstack([intrinsics_proj, np.zeros((3, 1))])
    
    print (f"Intrinsics projection matrix: {intrinsics_proj}")
    print (f"Extrinsics matrix: {extrinsics}")

    # Create video frames
    video_frames = []
    
    for t in tqdm(range(N), desc="Creating visualization"):
        # Get frame
        img = front_img[t]  # (H, W, 3)
        if img.max() <= 1.0:  # Normalize if needed
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
        
        # Get EE pose for this timestep and reshape for visualization
        ee_pose_t = ee_pose[t, None, :]  # (1, 6) - single pose point
        
        # Get EE action for this timestep - use chunked trajectory instead of single point
        ee_action_chunk_t = combined_xyz_act_cam[t]  # (100, 6) - full action trajectory
        
        # Draw end-effector pose as red dots
        img = draw_both_actions_on_frame(
            img,
            color="Reds",
            actions=ee_pose_t,
            arm="both",
            intrinsics=intrinsics_proj,
            extrinsics=extrinsics
        )
        
        # Draw end-effector action trajectory as green dots (visualize all 100 future steps)
        img = draw_both_actions_on_frame(
            img,
            color="Greens",
            actions=ee_action_chunk_t,  # Now passing (100, 6) instead of (1, 6)
            arm="both",
            intrinsics=intrinsics_proj,
            extrinsics=extrinsics
        )
        
        # Add frame info text
        cv2.putText(img, f'Frame {t}/{N}', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(img, f'Red: EE Pose (Left & Right)', (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(img, f'Green: EE Action Trajectory (Left & Right)', (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add EE position values as text
        left_pos = ee_pose[t, :3]
        right_pos = ee_pose[t, 3:]
        cv2.putText(img, f'Left EE: ({left_pos[0]:.2f}, {left_pos[1]:.2f}, {left_pos[2]:.2f})', 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(img, f'Right EE: ({right_pos[0]:.2f}, {right_pos[1]:.2f}, {right_pos[2]:.2f})', 
                   (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        video_frames.append(img)
    
    # Save video
    if video_frames:
        video_tensor = torch.stack([torch.from_numpy(frame) for frame in video_frames])
        episode_name = os.path.basename(os.path.dirname(episode_path))
        video_path = os.path.join(output_dir, f'{episode_name}_ee_visualization.mp4')
        print(f"Saving video: {video_path}")
        write_video_safe(video_path, video_tensor, fps=10)
        print(f"Video saved successfully!")
    else:
        print("No frames to save!")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize end-effector poses from a single LBM episode')
    parser.add_argument('--episode_path', type=str, required=True,
                        help='Path to the processed episode directory')
    parser.add_argument('--output_dir', type=str, default='./validation_videos',
                        help='Output directory for visualization video')
    parser.add_argument('--max_frames', type=int, default=None,
                        help='Maximum frames to process (default: all frames)')
    
    args = parser.parse_args()
    
    print(f"Episode path: {args.episode_path}")
    print(f"Output directory: {args.output_dir}")
    
    visualize_lbm_episode(
        args.episode_path,
        args.output_dir,
        args.max_frames
    )
    
    print("Visualization complete!")


if __name__ == "__main__":
    main()
