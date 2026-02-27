import os
import sys

# Add AVP python_client to path for tracker_pb2 import
avp_client_path = "/workspace/externals/EgoMimic/external/avp_teleop/python_client"
if avp_client_path not in sys.path:
    sys.path.insert(0, avp_client_path)

import numpy as np
from glob import glob
from tqdm import tqdm, trange
import time
import cv2 
from decord import VideoReader, cpu
import json 
import h5py
import gzip
import pickle
import shutil 
from multiprocessing import Pool
from scipy.spatial.transform import Rotation
from egomimic.utils.egomimicUtils import (
    ee_pose_to_cam_frame,
    interpolate_arr,
    cam_frame_to_cam_pixels,
)

"""
AVP Dataset Processing Script
Converts AVP (Action-Vision-Pose) dataset to EgoMimic HDF5 format

AVP data structure (per episode):
- episode.pkl (gzipped):
    - camera_intrinsics: list of (3, 3) camera intrinsics per frame
    - camera_extrinsics: list of (4, 4) camera extrinsics per frame
    - frame_timestamps: list of timestamps
    - pose_snapshots: list of dicts with 'left' and 'right' hand poses
    - success: bool
- main_camera.mp4: RGB video

Target EgoMimic HDF5 structure:
/data/demo_0/actions_xyz Dataset {N, 30, 6}
/data/demo_0/actions_xyz_act Dataset {N, 100, 6}
/data/demo_0/obs         Group
/data/demo_0/obs/ee_pose Dataset {N, 6}
/data/demo_0/obs/extrinsics Dataset {N, 4, 4}
/data/demo_0/obs/front_img_1 Dataset {N, 480, 640, 3}
/data/demo_0/obs/intrinsics Dataset {N, 3, 3}

Data:
AWS_REGION=us-east-1 aws s3 cp --recursive s3://robotics-manip-lbm/kylehatch/video_cotrain/HAMSTER_data/LBM_sim_egocentric/raw/data/tasks/egoPutKiwiInCenterOfTable egoPutKiwiInCenterOfTable
AWS_REGION=us-east-1 aws s3 cp --recursive s3://robotics-manip-lbm/kylehatch/video_cotrain/HAMSTER_data/LBM_sim_egocentric/raw/data/tasks/egoTurnMugRightsideUp egoTurnMugRightsideUp
AWS_REGION=us-east-1 aws s3 cp --recursive s3://robotics-manip-lbm/kylehatch/video_cotrain/HAMSTER_data/LBM_sim_egocentric/raw/data/tasks/egoTurnCupUpsideDown egoTurnCupUpsideDown

Usage:
docker exec swati-egomimic /bin/bash -c "cd /workspace/externals/EgoMimic && python egomimic/process_avp_to_egomimic.py --input_dir /workspace/externals/EgoMimic/datasets/AVP/raw --output_dir /workspace/externals/EgoMimic/datasets/AVP/processed/ --n_workers 8 --save_viz"

# Process a specific task only:
docker exec swati-egomimic /bin/bash -c "cd /workspace/externals/EgoMimic && python egomimic/process_avp_to_egomimic.py --input_dir /workspace/externals/EgoMimic/datasets/AVP/raw --output_dir /workspace/externals/EgoMimic/datasets/AVP/processed/ --task_name egoPutKiwiInCenterOfTable --n_workers 8"
"""
VAL_RATIO = 0.05

def split_train_val_from_hdf5(hdf5_path, val_ratio=VAL_RATIO):
    """Split dataset into train/val masks"""
    with h5py.File(hdf5_path, "a") as file:
        demo_keys = sorted([key for key in file["data"].keys() if "demo" in key])
        num_demos = len(demo_keys)
        num_val = int(np.ceil(num_demos * val_ratio))

        indices = np.arange(num_demos)
        np.random.shuffle(indices)

        val_indices = indices[:num_val]
        train_indices = indices[num_val:]

        train_mask = [demo_keys[i] for i in train_indices]
        val_mask = [demo_keys[i] for i in val_indices]

        file.create_dataset("mask/train", data=np.array(train_mask, dtype="S"))
        file.create_dataset("mask/valid", data=np.array(val_mask, dtype="S"))

def load_avp_episode(episode_pkl_path):
    """Load AVP episode data from pickle file"""
    try:
        with gzip.open(episode_pkl_path, 'rb') as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        print(f"Error loading {episode_pkl_path}: {e}")
        return None

def parse_to_se3(pose):
    """
    Parse translation and rotation data into SE(3) format (4x4 homogeneous matrix).
    Based on client_utils.py from AVP teleop code.
    
    Args:
        pose: Object with pose.translation and pose.rotation attributes,
              where each has x, y, z (and w for rotation) attributes
        
    Returns:
        se3_matrix: 4x4 numpy array representing the SE(3) transformation
    """
    # Extract translation values
    t = np.array([pose.translation.x, pose.translation.y, pose.translation.z])
    
    # Extract rotation values (quaternion as x, y, z, w)
    q = np.array([pose.rotation.x, pose.rotation.y, pose.rotation.z, pose.rotation.w])
    
    # Convert quaternion (x, y, z, w) to rotation matrix
    # scipy uses (x, y, z, w) format
    rot = Rotation.from_quat(q)
    R = rot.as_matrix()
    
    # Create SE(3) matrix (4x4)
    se3 = np.eye(4)
    se3[:3, :3] = R
    se3[:3, 3] = t
    
    return se3

def extract_hand_poses_from_response(response):
    """
    Extract hand pose from tracker response object.
    
    This extracts the hand anchor transform and optionally the index finger knuckle position.
    Based on visualize_collected_data.py from AVP teleop code.
    
    Returns SE(3) transformation matrix (4x4) representing the hand pose.
    """
    try:
        # Extract the anchor transform (base of the hand)
        if not hasattr(response, 'hand') or not hasattr(response.hand, 'anchor_transform'):
            print(f"[WARNING] Response missing hand.anchor_transform")
            return None
        
        anchor_transform = response.hand.anchor_transform
        anchor_transform_se3 = parse_to_se3(anchor_transform)
        
        # Optionally, we could also extract finger positions
        # For now, we'll use the anchor transform as the primary hand pose
        # This represents the wrist/palm position and orientation
        
        return anchor_transform_se3
        
    except Exception as e:
        print(f"[ERROR] Failed to extract hand pose: {e}")
        import traceback
        traceback.print_exc()
        return None

def process_single_episode(inpt):
    """Process a single AVP episode into EgoMimic format"""
    ep_no, episode_dir, step_size, im_dims, save_annotated_images = inpt 

    episode_data_list = {}
    
    # Create visualization directory for debugging
    viz_dir = None
    if ep_no == 0 or (ep_no <= 5 and save_annotated_images):
        viz_dir = f"./debug_viz_avp_ep_{ep_no}"
        os.makedirs(viz_dir, exist_ok=True)
        print(f"[Visualization] Will save debug frames to {viz_dir}")
    
    # Load episode data
    episode_pkl_path = os.path.join(episode_dir, "episode.pkl")
    video_path = os.path.join(episode_dir, "main_camera.mp4")
    
    if not os.path.exists(episode_pkl_path) or not os.path.exists(video_path):
        print(f"[Warning] Missing files for episode {ep_no} in {episode_dir}")
        return episode_data_list
    
    data = load_avp_episode(episode_pkl_path)
    if data is None:
        print(f"[Warning] Failed to load episode {ep_no}")
        return episode_data_list
    
    # Check if episode was successful
    if not data.get('success', False):
        print(f"[Warning] Episode {ep_no} marked as unsuccessful, skipping")
        return episode_data_list
    
    # Extract data
    camera_intrinsics_seq = data['camera_intrinsics']
    camera_extrinsics_seq = data['camera_extrinsics']
    frame_timestamps = data['frame_timestamps']
    pose_snapshots = data['pose_snapshots']
    
    N = len(frame_timestamps)
    
    # if ep_no == 0:
    #     print(f"[Debug] Episode has {N} frames")
    #     print(f"[Debug] Camera intrinsics[0]:\n{camera_intrinsics_seq[0]}")
    #     print(f"[Debug] Camera extrinsics[0]:\n{camera_extrinsics_seq[0]}")
    
    # Load video
    vr = VideoReader(video_path, ctx=cpu(0))
    if len(vr) != N:
        print(f"[Warning] Video frame count ({len(vr)}) doesn't match pose data ({N})")
        N = min(len(vr), N)
    
    # Get original image dimensions
    first_frame = vr[0].asnumpy()
    orig_im_h, orig_im_w = first_frame.shape[:2]
    
    if ep_no == 0:
        print(f"[Debug] Original image dimensions: {orig_im_w}x{orig_im_h}")
        print(f"[Debug] Target dimensions: {im_dims[0]}x{im_dims[1]}")
    
    # Scale intrinsics for resized images
    scale_x = im_dims[0] / orig_im_w
    scale_y = im_dims[1] / orig_im_h
    
    images = []
    ee_poses = []
    actions_xyz = []
    frame_indices = []
    intrinsics_seq = []
    extrinsics_seq = []
    
    # Extract hand poses from pose_snapshots
    left_hand_poses_3d = []
    right_hand_poses_3d = []
    device_extrinsics_list = []  # Store device poses for proper visualization
    
    for i in range(N):
        pose_snapshot = pose_snapshots[i]
        
        # Extract hand poses - check for None values
        if pose_snapshot.get('left') is None or pose_snapshot.get('right') is None:
            print(f"[Warning] Episode {ep_no} frame {i} has missing hand data, skipping episode")
            return {}
        
        left_response = pose_snapshot['left']['response']
        right_response = pose_snapshot['right']['response']
        
        left_pose = extract_hand_poses_from_response(left_response)
        right_pose = extract_hand_poses_from_response(right_response)
        
        if left_pose is None or right_pose is None:
            print(f"[ERROR] Cannot extract hand poses without proper tracker_pb2 module")
            print(f"[ERROR] You need to install/compile the tracker_pb2 protobuf module")
            print(f"[ERROR] Please check AVP dataset documentation for tracker_pb2 installation")
            return {}
        
        # Extract device extrinsics (world_T_device)
        if hasattr(left_response, 'device'):
            device_transform = parse_to_se3(left_response.device)
            device_extrinsics_list.append(device_transform)
        else:
            # Fallback to identity if no device info
            device_extrinsics_list.append(np.eye(4))
        
        left_hand_poses_3d.append(left_pose)
        right_hand_poses_3d.append(right_pose)
    
    left_hand_poses_3d = np.array(left_hand_poses_3d)  # (N, 4, 4)
    right_hand_poses_3d = np.array(right_hand_poses_3d)  # (N, 4, 4)
    device_extrinsics_list = np.array(device_extrinsics_list)  # (N, 4, 4)
    
    # Extract translations for action generation (N, 3)
    left_hand_positions = left_hand_poses_3d[:, :3, 3]
    right_hand_positions = right_hand_poses_3d[:, :3, 3]
    
    # Process frames and create action chunks
    for i in range(0, N, 1):  # Process every frame
        # Read and resize frame
        full_res_image = vr[i].asnumpy()
        initial_image = cv2.resize(full_res_image, im_dims)
        
        # Scale intrinsics and correct extrinsics (used for both viz and saving)
        scaled_intrinsics = camera_intrinsics_seq[i].copy()
        scaled_intrinsics[0, 0] *= scale_x  # fx
        scaled_intrinsics[1, 1] *= scale_y  # fy
        scaled_intrinsics[0, 2] *= scale_x  # cx
        scaled_intrinsics[1, 2] *= scale_y  # cy
        
        # Apply the correct extrinsics transformation
        device_T_camera = camera_extrinsics_seq[i].T
        device_T_camera_corrected = np.linalg.inv(device_T_camera)
        world_T_camera = device_extrinsics_list[i] @ device_T_camera_corrected
        
        # Store for saving
        intrinsics_seq.append(scaled_intrinsics)
        extrinsics_seq.append(world_T_camera)
        
        # Create action chunk (future trajectory)
        end_idx = min(i + step_size, N)
        
        # Get future hand positions (using positions, not full SE3)
        left_future = left_hand_positions[i:end_idx]
        right_future = right_hand_positions[i:end_idx]
        
        # Pad if necessary
        if len(left_future) < step_size:
            pad_len = step_size - len(left_future)
            left_future = np.vstack([left_future, np.tile(left_future[-1], (pad_len, 1))])
            right_future = np.vstack([right_future, np.tile(right_future[-1], (pad_len, 1))])
        
        # Transform to camera frame at current timestep using corrected extrinsics
        left_actions = ee_pose_to_cam_frame(left_future, world_T_camera)
        right_actions = ee_pose_to_cam_frame(right_future, world_T_camera)
        hand_actions = np.hstack((left_actions, right_actions))  # (step_size, 6)
        
        actions_xyz.append(hand_actions)
        images.append(initial_image)
        frame_indices.append(i)
        
        # Current EE pose (using positions) with corrected extrinsics
        left_ee = ee_pose_to_cam_frame(
            left_hand_positions[i:i+1], 
            world_T_camera
        )
        right_ee = ee_pose_to_cam_frame(
            right_hand_positions[i:i+1],
            world_T_camera
        )
        ee_pose = np.hstack((left_ee, right_ee))  # (1, 6)
        ee_poses.append(ee_pose.squeeze())

        if viz_dir is not None:
            viz_image = initial_image.copy()

            left_2d_arr = cam_frame_to_cam_pixels(left_ee, scaled_intrinsics)[0]
            right_2d_arr = cam_frame_to_cam_pixels(right_ee, scaled_intrinsics)[0]
            # print (left_2d_arr.shape, right_2d_arr.shape)
            # import ipdb; ipdb.set_trace()
            
            # Convert to integer tuples for cv2
            left_2d = (int(left_2d_arr[0]), int(left_2d_arr[1]))
            right_2d = (int(right_2d_arr[0]), int(right_2d_arr[1]))
            
            # Draw on image
            if 0 <= left_2d[0] < im_dims[0] and 0 <= left_2d[1] < im_dims[1]:
                cv2.circle(viz_image, left_2d, 25, (0, 255, 0), -1)
                cv2.circle(viz_image, left_2d, 27, (255, 255, 255), 3)
                cv2.putText(viz_image, "L", (left_2d[0] - 20, left_2d[1] - 35), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)
            
            if 0 <= right_2d[0] < im_dims[0] and 0 <= right_2d[1] < im_dims[1]:
                cv2.circle(viz_image, right_2d, 25, (255, 0, 0), -1)
                cv2.circle(viz_image, right_2d, 27, (255, 255, 255), 3)
                cv2.putText(viz_image, "R", (right_2d[0] - 20, right_2d[1] - 35), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 4)
            
            # Add frame info
            cv2.putText(viz_image, f"Frame: {i}/{N}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Save visualization frame
            viz_path = os.path.join(viz_dir, f"frame_{i:04d}.png")
            cv2.imwrite(viz_path, cv2.cvtColor(viz_image, cv2.COLOR_RGB2BGR))

    
    if len(images) == 0:
        print(f"[Skipping] No valid data for episode {ep_no}")
        return episode_data_list
    
    # Convert to arrays
    actions_xyz = np.array(actions_xyz)  # (seq_len, step_size, 6)
    actions_xyz_act = interpolate_arr(actions_xyz, 100)  # (seq_len, 100, 6)
    intrinsics_seq = np.array(intrinsics_seq)  # (seq_len, 3, 3)
    extrinsics_seq = np.array(extrinsics_seq)  # (seq_len, 4, 4)
    
    # Return processed data
    episode_data_list = {
        "ep_no": ep_no,
        "actions_xyz": actions_xyz,
        "actions_xyz_act": actions_xyz_act,
        "front_img_1": np.array(images),
        "ee_pose": np.array(ee_poses),
        "num_samples": int(actions_xyz.shape[0]),
        "intrinsics": intrinsics_seq,
        "extrinsics": extrinsics_seq
    }
    
    return episode_data_list

def process_task(task_dir, output_hdf5_path, n_workers, step_size, im_dims, 
                 save_annotated_images):
    """Process all episodes in a task directory"""
    
    # Find all episode directories
    episode_dirs = []
    for session_dir in glob(os.path.join(task_dir, "*")):
        if not os.path.isdir(session_dir):
            continue
        for ep_dir in sorted(glob(os.path.join(session_dir, "episode*"))):
            if os.path.isdir(ep_dir):
                episode_dirs.append(ep_dir)
    
    print(f"Found {len(episode_dirs)} episodes in {task_dir}")
    
    if len(episode_dirs) == 0:
        print(f"[SKIP] No episodes found, NOT creating HDF5 file")
        return 0
    
    # Prepare function inputs
    function_inpts = []
    for ep_no, ep_dir in enumerate(episode_dirs):
        function_inpts.append((ep_no, ep_dir, step_size, 
                              im_dims, save_annotated_images))
    
    # Process episodes with multiprocessing
    batch_size = n_workers * 2
    total_successful = 0
    demo_idx = 0
    
    # Only initialize HDF5 file AFTER we know we have episodes to process
    hdf5_initialized = False
    
    for batch_start in range(0, len(function_inpts), batch_size):
        batch_end = min(batch_start + batch_size, len(function_inpts))
        batch_inpts = function_inpts[batch_start:batch_end]
        
        print(f"Processing batch {batch_start//batch_size + 1}: "
              f"episodes {batch_start} to {batch_end-1}")
        
        # Process batch
        with Pool(n_workers) as p:
            batch_results = list(p.imap(process_single_episode, batch_inpts))
        
        # Filter successful episodes
        successful_batch = [result for result in batch_results if len(result) > 0]
        
        # Write to HDF5 - only create file when we have successful episodes
        if successful_batch:
            # Initialize HDF5 file on first successful batch
            if not hdf5_initialized:
                with h5py.File(output_hdf5_path, "w") as f:
                    f.create_group("data")
                hdf5_initialized = True
            
            with h5py.File(output_hdf5_path, "a") as f:
                data = f["data"]
                
                for episode_data in successful_batch:
                    group = data.create_group(f"demo_{demo_idx}")
                    group.create_dataset("actions_xyz", data=episode_data["actions_xyz"])
                    group.create_dataset("actions_xyz_act", data=episode_data["actions_xyz_act"])
                    group.create_dataset("obs/front_img_1", data=episode_data["front_img_1"])
                    group.create_dataset("obs/ee_pose", data=episode_data["ee_pose"])
                    group.create_dataset("obs/intrinsics", data=episode_data["intrinsics"])
                    group.create_dataset("obs/extrinsics", data=episode_data["extrinsics"])
                    group.attrs["num_samples"] = episode_data["num_samples"]
                    demo_idx += 1
        
        batch_successful = len(successful_batch)
        batch_failed = len(batch_results) - batch_successful
        total_successful += batch_successful
        
        print(f"Batch complete: {batch_successful} successful, {batch_failed} failed")
    
    return total_successful

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process AVP dataset to EgoMimic HDF5 format")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Path to the AVP task directory (e.g., /path/to/AVP/egoPutKiwiInCenterOfTable)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Path to output directory where HDF5 file will be saved")
    parser.add_argument("--step_size", type=int, default=30,
                        help="Action horizon/chunk size (default: 30)")
    parser.add_argument("--image_width", type=int, default=640,
                        help="Output image width (default: 640)")
    parser.add_argument("--image_height", type=int, default=480,
                        help="Output image height (default: 480)")
    parser.add_argument("--n_workers", type=int, default=8,
                        help="Number of parallel workers (default: 8)")
    parser.add_argument("--val_ratio", type=float, default=VAL_RATIO,
                        help="Validation split ratio (default: 0.05)")
    parser.add_argument("--save_viz", action="store_true",
                        help="Save visualization frames for debugging")
    parser.add_argument("--task_name", type=str, default=None,
                        help="Specific task name to process (e.g., egoPutKiwiInCenterOfTable). If not provided, processes all tasks.")
    
    args = parser.parse_args()
    
    # Configuration from arguments
    STEP_SIZE = args.step_size
    NEW_IMAGE_W = args.image_width
    NEW_IMAGE_H = args.image_height
    SAVE_ANNOTATED_IMAGES = args.save_viz
    N_WORKERS = args.n_workers
    
    im_dims = np.array([NEW_IMAGE_W, NEW_IMAGE_H])
    
    # Paths
    task_dir = args.input_dir
    output_dir = args.output_dir
    
    # Get task name from input directory and create output HDF5 path
    tasks_to_process = os.listdir(task_dir)
    
    # Filter to specific task if --task_name is provided
    if args.task_name:
        if args.task_name in tasks_to_process:
            tasks_to_process = [args.task_name]
            print(f"[INFO] Filtering to single task: {args.task_name}")
        else:
            print(f"[ERROR] Task '{args.task_name}' not found in {task_dir}")
            print(f"Available tasks: {tasks_to_process}")
            sys.exit(1)
    
    # Filter to only directories
    tasks_to_process = [t for t in tasks_to_process if os.path.isdir(os.path.join(task_dir, t))]
    print(f"[INFO] Tasks to process: {tasks_to_process}")
    
    for task in tasks_to_process:
        task_path = os.path.join(task_dir, task)
        task_name = os.path.basename(task_path.rstrip('/'))
        os.makedirs(output_dir, exist_ok=True)
        output_hdf5 = os.path.join(output_dir, f"{task_name}.hdf5")
        
        print(f"\n{'='*60}")
        print(f"Processing task: {task_name}")
        print(f"Input directory: {task_path}")
        print(f"Output HDF5: {output_hdf5}")
        print(f"{'='*60}\n")
        
        # Process task
        num_successful = process_task(
            task_dir=task_path,
            output_hdf5_path=output_hdf5,
            n_workers=N_WORKERS,
            step_size=STEP_SIZE,
            im_dims=im_dims,
            save_annotated_images=SAVE_ANNOTATED_IMAGES,
        )
        
        if num_successful > 0:
            # Add train/val split
            split_train_val_from_hdf5(output_hdf5, val_ratio=args.val_ratio)
            print(f"\n{'='*60}")
            print(f"Successfully processed {num_successful} episodes")
            print(f"Output saved to: {output_hdf5}")
            print(f"Train/val split: {1-args.val_ratio:.0%} / {args.val_ratio:.0%}")
            print(f"{'='*60}\n")
        else:
            print(f"\n[ERROR] No episodes were successfully processed for {task_name}!")
            print(f"Please check the tracker_pb2 installation and hand pose extraction.")
            # Don't try to remove - HDF5 was never created if num_successful == 0

"""
Usage (inside docker):
    # Process all tasks:
    python egomimic/process_avp_to_egomimic.py \
        --input_dir /workspace/externals/EgoMimic/datasets/AVP/raw/ \
        --output_dir /workspace/externals/EgoMimic/datasets/AVP/processed \
        --n_workers 8 \
        --val_ratio 0.05

    # Process a specific task:
    python egomimic/process_avp_to_egomimic.py \
        --input_dir /workspace/externals/EgoMimic/datasets/AVP/raw/ \
        --output_dir /workspace/externals/EgoMimic/datasets/AVP/processed \
        --task_name egoPutKiwiInCenterOfTable \
        --n_workers 8

    This will create: /workspace/externals/EgoMimic/datasets/AVP/processed/egoPutKiwiInCenterOfTable.hdf5

Optional arguments:
    --task_name "taskname"      Process only this specific task (default: all tasks)
    --step_size 30              Action horizon/chunk size
    --image_width 640           Output image width
    --image_height 480          Output image height
    --n_workers 8               Number of parallel workers
    --val_ratio 0.05            Validation split ratio (0.0 to 1.0)
    --save_viz                  Save visualization frames for debugging

Note: The HDF5 filename is automatically generated from the task directory name.
      This script requires the tracker_pb2 module for extracting hand poses.
"""
