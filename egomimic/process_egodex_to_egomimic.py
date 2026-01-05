import os
import sys
import numpy as np
from glob import glob
from tqdm import tqdm, trange
import time
import cv2 
# import imageio
# from collections import defaultdict
from decord import VideoReader, cpu
# from torchcodec.decoders import VideoDecoder
import json 
import yaml 
import h5py
import boto3
from botocore.exceptions import ClientError
import shutil 
import subprocess
from urllib.parse import urlparse
from multiprocessing import Pool
# from rdp import rdp
# import random
from egomimic.utils.egomimicUtils import (
    ee_pose_to_cam_frame,
    interpolate_arr,
)
"""
/data/demo_0/actions_xyz Dataset {300, 30, 6}
/data/demo_0/actions_xyz_act Dataset {300, 100, 6}
/data/demo_0/obs         Group
/data/demo_0/obs/ee_pose Dataset {300, 6}
/data/demo_0/obs/extrinsics Dataset {300, 4, 4}
/data/demo_0/obs/front_img_1 Dataset {300, 480, 640, 3}
/data/demo_0/obs/intrinsics Dataset {300, 3, 3}

"""
def split_train_val_from_hdf5(hdf5_path, val_ratio):
    with h5py.File(hdf5_path, "a") as file:
        demo_keys = [key for key in file["data"].keys() if "demo" in key]
        num_demos = len(demo_keys)
        num_val = int(np.ceil(num_demos * val_ratio))

        indices = np.arange(num_demos)
        np.random.shuffle(indices)

        val_indices = indices[:num_val]
        train_indices = indices[num_val:]

        train_mask = [f"demo_{i}" for i in train_indices]
        val_mask = [f"demo_{i}" for i in val_indices]

        file.create_dataset("mask/train", data=np.array(train_mask, dtype="S"))
        file.create_dataset("mask/valid", data=np.array(val_mask, dtype="S"))

def load_json(json_file):
    with open(json_file) as f:
        obj = json.load(f)
    return obj

def save_to_json(file_path, obj):
    with open(file_path, 'w') as f:
        json.dump(obj , f, indent='\t')

def list_s3_subdirectories(s3_uri):
    parsed = urlparse(s3_uri)
    bucket_name = parsed.netloc
    prefix = parsed.path.strip('/') + "/"

    s3 = boto3.client('s3')
    paginator = s3.get_paginator('list_objects_v2')
    result = paginator.paginate(
        Bucket=bucket_name,
        Prefix=prefix,
        Delimiter='/'
    )

    subdirs = []
    for page in result:
        if 'CommonPrefixes' in page:
            for cp in page['CommonPrefixes']:
                subdirs.append(cp['Prefix'])
    return subdirs

def s3_file_exists(s3_uri):
    parsed = urlparse(s3_uri)
    bucket_name = parsed.netloc
    object_key = parsed.path.strip('/')

    s3 = boto3.client('s3')
    try:
        s3.head_object(Bucket=bucket_name, Key=object_key)
        return True
    except ClientError as e:
        if e.response['Error']['Code'] == "404":
            return False
        else:
            raise  # Something else went wrong

def download_from_s3(s3_path, local_path):
    # command = ["aws", "s3", s3_command, s3_path, local_path]
    s3_path = os.path.join(s3_path, "*")
    command = ["s5cmd", "cp", "--concurrency=4", s3_path, local_path]
    print (command)

    print(f"Downloading {s3_path} to {local_path}...")
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    # process = subprocess.Popen(command)
    stdout, stderr = process.communicate()

    if process.returncode == 0:
        # print("Download successful!")
        return True
    else:
        print(f"Error downloading folder:\n{stderr}")
        return False

def upload_to_s3(local_path, s3_path):
    # Use s5cmd for file uploads
    if os.path.isfile(local_path):
        command = ["s5cmd", "cp", "--concurrency=4", local_path, s3_path]

    print(f"Uploading {local_path} to {s3_path}...")
    print (command)

    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    # process = subprocess.Popen(command)
    stdout, stderr = process.communicate()

    if process.returncode == 0:
        # print("Upload successful!")
        return True
    else:
        print(f"Error uploading:\n{stderr}")
        return False

# Egodex data arrangement
# part1
# └──task1
#     └──0.hdf5
#     └──0.mp4
# EGODEX hdf5 data structure
# If the corresponding MP4 file is T seconds long, then N = 30 * T
# camera
# └──intrinsic            # 3 x 3 camera intrinsics. always the same in every file.

# transforms              # all joint transforms, all below have shape N x 4 x 4.
# └──camera               
# └──leftHand             
# └──rightHand            
# └──leftIndexFingerTip
# └──leftIndexFingerKnuckle
# └──(64 more joints...)

# confidences             # (optional) scalar joint confidences, all below have shape N.
# └──leftHand
# └──rightHand
# └──(66 more joints...)
# Note that leftHand and rightHand refer to the wrists in egodex (end effector)

def process_single_episode(inpt):
    ep_no, mp4_file, hdf5_file, step_size, oversample_rate, hand_detection_confidence_threshold, im_dims, save_annotated_images = inpt 

    episode_data_list = {}
    left_keypoint_name = "leftIndexFingerKnuckle"  # Using index finger knuckle as EE pose
    right_keypoint_name = "rightIndexFingerKnuckle"  # Using index finger knuckle as EE pose
    # Create visualization directory for first demo only
    viz_dir = None
    if ep_no == 0 or (ep_no <= 5 and save_annotated_images):  # Visualize first demo or first 5 if debug mode
        viz_dir = f"./debug_viz_ep_{ep_no}"
        os.makedirs(viz_dir, exist_ok=True)
        print(f"[Visualization] Saving debug frames to {viz_dir}")
    
    with h5py.File(hdf5_file, 'r') as f:
        # f.keys(): <KeysViewHDF5 ['camera', 'confidences', 'transforms']>
        camera_extrinsics =  f["transforms"]["camera"][:]  # (300, 4, 4)
        left_idxfinger_knuckle_3D_poses = f["transforms"][left_keypoint_name][:]  # (300, 4, 4)
        right_idxfinger_knuckle_3D_poses = f["transforms"][right_keypoint_name][:]

        if "confidences" in f:
            left_idxfinger_knuckle_confidences = f["confidences"][left_keypoint_name][:] # (300, )
            right_idxfinger_knuckle_confidences = f["confidences"][right_keypoint_name][:]
        else:
            print (f"[Warning] No confidence data found in hdf5 for episode {ep_no}, skipping episode")
            return episode_data_list

        if left_idxfinger_knuckle_confidences.max() < hand_detection_confidence_threshold or right_idxfinger_knuckle_confidences.max() < hand_detection_confidence_threshold:
            print (f"[Warning] all hand tracks detected below confidence threshold {hand_detection_confidence_threshold} for episode {ep_no}, skipping episode")
            # skip
            return episode_data_list

        N = camera_extrinsics.shape[0]
        
        # Get intrinsics for visualization
        camera_intrinsics = f["camera/intrinsic"][:]  # (3, 3)
    
        # decoder = VideoDecoder(mp4_file, device="cpu")
        vr = VideoReader(mp4_file, ctx=cpu(0))
        assert len(vr) == N, f"decoder.metadata.num_frames: {len(vr)}, N: {N}"
        # print ("Seq len: ", N, step_size, decoder.metadata.num_frames)
        
        # Get original image dimensions from first frame
        first_frame = vr[0].asnumpy()
        orig_im_h, orig_im_w = first_frame.shape[:2]  # (H, W, C)
        # original_im_dims = np.array([decoder.metadata.width, decoder.metadata.height])
        
        if ep_no == 0:
            print(f"[Debug] Original image dimensions: {orig_im_w}x{orig_im_h} (W x H)")
            print(f"[Debug] Target im_dims: {im_dims} [W, H]")
            print(f"[Debug] Camera intrinsics (original):\n{camera_intrinsics}")
        
        # Scale intrinsics for resized images
        scale_x = im_dims[0] / orig_im_w
        scale_y = im_dims[1] / orig_im_h
        viz_intrinsics = camera_intrinsics.copy()
        viz_intrinsics[0, 0] *= scale_x  # fx
        viz_intrinsics[1, 1] *= scale_y  # fy
        viz_intrinsics[0, 2] *= scale_x  # cx
        viz_intrinsics[1, 2] *= scale_y  # cy
        
        if ep_no == 0:
            print(f"[Debug] Scale factors: scale_x={scale_x:.4f}, scale_y={scale_y:.4f}")
            print(f"[Debug] Scaled intrinsics for visualization:\n{viz_intrinsics}")
   
        images = []
        ee_poses = []
        actions_xyz = []
        actions_xyz_act = []  # interpolated actions
        frame_indices = []  # Track which frames were actually used

        
        # Check if running in interactive terminal for progress bars
        use_tqdm = False
        iterator = trange(0, len(vr), desc=f"[Processing episode {ep_no}]") if use_tqdm else range(0, len(vr))
        
        for i in iterator:
            # Decord VideoReader already returns (H, W, C) format
            full_res_image = vr[i].asnumpy()  # (H, W, C) - original resolution
            # print (f"initial_image.shape: {initial_image.shape}")
            initial_image = cv2.resize(full_res_image, im_dims)  # Resize for processing
            
            # Visualize for debugging (first demo only or when save_annotated_images is True)
            if viz_dir is not None:
                # Use resized image for visualization
                viz_image = initial_image.copy()
                
                # Project left and right hand poses to image
                # Extract translation from 4x4 transform matrix: [:3, 3] not [3, :3]!
                left_hand_3d = left_idxfinger_knuckle_3D_poses[i][:3, 3].reshape(1, -1)  # (1, 3)
                right_hand_3d = right_idxfinger_knuckle_3D_poses[i][:3, 3].reshape(1, -1)  # (1, 3)
                
                # Transform to camera frame
                left_hand_cam = ee_pose_to_cam_frame(left_hand_3d, camera_extrinsics[i])  # (1, 3)
                right_hand_cam = ee_pose_to_cam_frame(right_hand_3d, camera_extrinsics[i])  # (1, 3)
                
                # Debug: print 3D coordinates
                if i == 0:
                    print(f"[Debug] Left hand 3D (world): {left_hand_3d}")
                    print(f"[Debug] Right hand 3D (world): {right_hand_3d}")
                    print(f"[Debug] Left hand 3D (camera frame): {left_hand_cam}")
                    print(f"[Debug] Right hand 3D (camera frame): {right_hand_cam}")
                
                # Project to 2D using scaled intrinsics
                def project_to_2d(point_3d, K):
                    """Project 3D point in camera frame to 2D image coordinates."""
                    x, y, z = point_3d[0]
                    if z > 0:  # Only project points in front of camera
                        u = int(K[0, 0] * x / z + K[0, 2])
                        v = int(K[1, 1] * y / z + K[1, 2])
                        return (u, v), True
                    return (0, 0), False
                
                left_2d, left_valid = project_to_2d(left_hand_cam, viz_intrinsics)
                right_2d, right_valid = project_to_2d(right_hand_cam, viz_intrinsics)
                
                # Debug: print image dimensions and coordinates
                if i == 0:
                    print(f"[Debug] Resized viz_image.shape: {viz_image.shape}")
                    print(f"[Debug] Left 2D: {left_2d}, Right 2D: {right_2d}")
                    print(f"[Debug] Using scaled intrinsics for projection")
                
                # Draw circles on the image - large and very visible
                if left_valid:
                    in_bounds = 0 <= left_2d[0] < im_dims[0] and 0 <= left_2d[1] < im_dims[1]
                    color = (0, 255, 0) if in_bounds else (0, 255, 255)  # Green if in bounds, Yellow if out
                    cv2.circle(viz_image, left_2d, 25, color, -1)
                    cv2.circle(viz_image, left_2d, 27, (255, 255, 255), 3)
                    cv2.putText(viz_image, "L", (left_2d[0] - 20, left_2d[1] - 35), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 4)
                    if not in_bounds:
                        print(f"[Warning] Left hand out of bounds at frame {i} for episode {ep_no}: {left_2d}, im_dims: {im_dims}")
                
                if right_valid:
                    in_bounds = 0 <= right_2d[0] < im_dims[0] and 0 <= right_2d[1] < im_dims[1]
                    color = (255, 0, 0) if in_bounds else (255, 255, 0)  # Blue if in bounds, Yellow if out
                    cv2.circle(viz_image, right_2d, 25, color, -1)
                    cv2.circle(viz_image, right_2d, 27, (255, 255, 255), 3)
                    cv2.putText(viz_image, "R", (right_2d[0] - 20, right_2d[1] - 35), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 4)
                    if not in_bounds:
                        print(f"[Warning] Right hand out of bounds at frame {i} for episode {ep_no}: {right_2d}, im_dims: {im_dims}")

                
                # Add frame info
                cv2.putText(viz_image, f"Frame: {i}/{len(vr)}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Save every 10th frame to avoid too many files
                if i % 10 == 0:
                    cv2.imwrite(f"{viz_dir}/frame_{i:04d}.jpg", cv2.cvtColor(viz_image, cv2.COLOR_RGB2BGR))
            
            
            # initialize segments
            left_idxfinger_knuckle_3D_poses_segment = np.zeros((step_size, 4, 4))
            right_idxfinger_knuckle_3D_poses_segment = np.zeros((step_size, 4, 4))
            left_idxfinger_knuckle_confidences_segment = np.zeros((step_size, ))
            right_idxfinger_knuckle_confidences_segment = np.zeros((step_size, ))

            if i + step_size > N:
                # Copy-Padding for the last few frames
                left_idxfinger_knuckle_3D_poses_segment[:N-i] = left_idxfinger_knuckle_3D_poses[i:N]
                left_idxfinger_knuckle_3D_poses_segment[N-i:] = np.tile(left_idxfinger_knuckle_3D_poses[N-1], (i+step_size-N, 1, 1))
                right_idxfinger_knuckle_3D_poses_segment[:N-i] = right_idxfinger_knuckle_3D_poses[i:N]
                right_idxfinger_knuckle_3D_poses_segment[N-i:] = np.tile(right_idxfinger_knuckle_3D_poses[N-1], (i+step_size-N, 1, 1))
                if "confidences" in f:
                    left_idxfinger_knuckle_confidences_segment[:N-i] = left_idxfinger_knuckle_confidences[i:N]
                    left_idxfinger_knuckle_confidences_segment[N-i:] = np.tile(left_idxfinger_knuckle_confidences[N-1], (i+step_size-N,))
                    right_idxfinger_knuckle_confidences_segment[:N-i] = right_idxfinger_knuckle_confidences[i:N]
                    right_idxfinger_knuckle_confidences_segment[N-i:] = np.tile(right_idxfinger_knuckle_confidences[N-1], (i+step_size-N,))
            else:
                left_idxfinger_knuckle_3D_poses_segment = left_idxfinger_knuckle_3D_poses[i:i+step_size] # (30, 4, 4)
                right_idxfinger_knuckle_3D_poses_segment = right_idxfinger_knuckle_3D_poses[i:i+step_size]
                if "confidences" in f:
                    left_idxfinger_knuckle_confidences_segment = left_idxfinger_knuckle_confidences[i:i+step_size] #(30, )
                    right_idxfinger_knuckle_confidences_segment = right_idxfinger_knuckle_confidences[i:i+step_size]

            path_segments = {"left":{"3D_poses":left_idxfinger_knuckle_3D_poses_segment},
                                "right":{"3D_poses":right_idxfinger_knuckle_3D_poses_segment}}
            
            
            if "confidences" in f:
                path_segments["left"]["confidences"] = left_idxfinger_knuckle_confidences_segment
                path_segments["right"]["confidences"] = right_idxfinger_knuckle_confidences_segment
            
            # Get indices of frames where confidence is above threshold
            valid_indices = set()
            for handside in ["left", "right"]:
                if "confidences" in path_segments[handside]:
                    pose_segment_confidences = path_segments[handside]["confidences"]
                    valid_indices_handside = np.where(pose_segment_confidences > hand_detection_confidence_threshold)[0]
                    valid_indices.update(valid_indices_handside.tolist())

            valid_indices = sorted(list(valid_indices))
            # print (f"valid_indices: {valid_indices}, len: {len(valid_indices)}")
            if len(valid_indices) == step_size:
                pass
                # print (f"[All valid] All hand tracks detected above confidence threshold {hand_detection_confidence_threshold} for episode {ep_no}, step starting at {i}")
                # for these valid indices, pick out the hand tracks -  actions, interp actions
            else:
                # print (f"[Partial valid] {len(valid_indices)}/{step_size} hand tracks detected above confidence threshold {hand_detection_confidence_threshold} for episode {ep_no}, step starting at {i}, skipping this chunk")
                continue

            # Our goal is to construct the action-chunk (a_t -> a_t+H) by transforming each position in the trajectory into the
            # observation camera frame Ft
            # Extract translation from 4x4 transform matrix: [:3, 3] not [3, :3]!
            left_hand_actions = ee_pose_to_cam_frame(path_segments["left"]["3D_poses"][valid_indices, :3, 3], camera_extrinsics[i])  # (num_valid, 3) pick translation component from matrix
            right_hand_actions = ee_pose_to_cam_frame(path_segments["right"]["3D_poses"][valid_indices, :3, 3], camera_extrinsics[i])
            hand_actions = np.hstack((left_hand_actions, right_hand_actions))  # (num_valid, 6)
            
            actions_xyz.append(hand_actions)
            images.append(initial_image)
            frame_indices.append(i)  # Track this frame index
            # Convert EE pose also to camera frame Ft
            # Extract translation from 4x4 transform matrix: [:3, 3] not [3, :3]!
            left_ee_pose = ee_pose_to_cam_frame(left_idxfinger_knuckle_3D_poses[i][:3, 3].reshape(1, -1), camera_extrinsics[i])
            right_ee_pose = ee_pose_to_cam_frame(right_idxfinger_knuckle_3D_poses[i][:3, 3].reshape(1, -1), camera_extrinsics[i])
            stacked_ee_pose = np.hstack((left_ee_pose, right_ee_pose))  # at current time step # (6, )
            ee_poses.append(stacked_ee_pose)

        
        if len(images) == 0:
            print (f"[Skipping] No valid data for episode {ep_no}")
            return episode_data_list
        
        # print (f"len(images): {len(images)}, len(ee_poses): {len(ee_poses)}, len(actions_xyz): {len(actions_xyz)}, actions_xyz[0].shape: {actions_xyz[0].shape}, actions_xyz[-1].shape: {actions_xyz[-1].shape}")

        actions_xyz = np.array(actions_xyz)
        actions_xyz_act = interpolate_arr(actions_xyz, 100)  # (num_valid * oversample_rate, 6)
        
        # Get sequence length
        seq_len = actions_xyz.shape[0]
        
        # Repeat intrinsics for each timestep in the sequence to match other observations
        # This ensures intrinsics have the same first dimension as extrinsics and other data
        intrinsics_seq = np.tile(viz_intrinsics[np.newaxis, :, :], (seq_len, 1, 1))  # (seq_len, 3, 3)
        
        # Extract only the extrinsics for frames that were actually used
        extrinsics_seq = camera_extrinsics[frame_indices]  # (seq_len, 4, 4)
        
        # Return processed data instead of writing directly
        episode_data_list = {
            "ep_no": ep_no,
            "actions_xyz": actions_xyz,
            "actions_xyz_act": actions_xyz_act,
            "front_img_1": np.array(images),
            "ee_pose": np.squeeze(np.array(ee_poses)),
            "num_samples": int(actions_xyz.shape[0]),
            "intrinsics": intrinsics_seq,
            "extrinsics": extrinsics_seq
        }
        
        return episode_data_list

    return {}

def process_data_into_egodex_format(local_download_dir, 
                                     local_processed_dir, 
                                     desc,
                                     n_workers, 
                                     step_size, 
                                     im_dims,
                                     save_annotated_images, 
                                     oversample_rate, 
                                     singlehand,
                                     hand_detection_confidence_threshold,
                                     hdf5_write_path,
                                     batch_size=None):

    # ### DEBUG ###
    print (f"Processing {local_download_dir} into {local_processed_dir} with {n_workers} workers")
    # hdf5_files = glob(os.path.join(local_download_dir, "*.hdf5"))
    # episode_numbers = sorted([int(hdf5_file.split("/")[-1].split(".")[0]) for hdf5_file in hdf5_files])
    # for ep_no in episode_numbers:
    #     hdf5_file = os.path.join(local_download_dir, f"{ep_no}.hdf5")

    #     with h5py.File(hdf5_file, 'r') as f:
    #         camera_extrinsics =  f["transforms"]["camera"][:]
    #         llm_description = f.attrs['llm_description']
    #         llm_description2 = f.attrs['llm_description2']

    #         print(f"\nep_no: {ep_no}, camera_extrinsics.shape[0]: {camera_extrinsics.shape[0]}")
    #         print(f"\tllm_description: {llm_description}")
    #         print(f"\tllm_description2: {llm_description2}")

    # return []
    # ### END DEBUG ###

    mp4_files = glob(os.path.join(local_download_dir, "*.mp4"))
    episode_numbers = sorted([int(mp4_file.split("/")[-1].split(".")[0]) for mp4_file in mp4_files])

    # Set batch size (number of episodes to process before writing to HDF5)
    if batch_size is None:
        batch_size = n_workers * 2  # Process 2 batches per worker set
    
    total_successful = 0
    demo_idx = 0
    
    # Process episodes in batches
    for batch_start in range(0, len(episode_numbers), batch_size):
        batch_end = min(batch_start + batch_size, len(episode_numbers))
        batch_episodes = episode_numbers[batch_start:batch_end]
        
        print(f"Processing batch {batch_start//batch_size + 1}: episodes {batch_start} to {batch_end-1}")
        
        function_inpts = []
        # DEBUG: Reduce no of demo episodes processed for testing
        # batch_episodes = batch_episodes[:1]
        for ep_no in batch_episodes:
            mp4_file = os.path.join(local_download_dir, f"{ep_no}.mp4")
            hdf5_file = os.path.join(local_download_dir, f"{ep_no}.hdf5")
            function_inpts.append((ep_no, mp4_file, hdf5_file, step_size, oversample_rate, hand_detection_confidence_threshold, im_dims, save_annotated_images))

        # Process batch with multiprocessing
        with Pool(n_workers) as p:
            batch_results = list(p.imap(process_single_episode, function_inpts))
        
        # Filter successful episodes
        successful_batch = [result for result in batch_results if len(result) > 0]
        
        # Write successful episodes to HDF5 sequentially (no race conditions)
        if successful_batch:
            with h5py.File(hdf5_write_path, "a") as f:
                if "data" not in f:
                    data = f.create_group("data")
                else:
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
        
        # Optional: force garbage collection after each batch to free memory
        import gc
        gc.collect()
    
    print(f"Processed {total_successful} episodes successfully for task in {local_download_dir}")
    return total_successful

def load_from_task_list(task_list_path):
    with open(task_list_path, 'r') as f:
        tasks_subset = f.read().splitlines()
    return tasks_subset

if __name__ == "__main__":
    
    STEP_SIZE = 30 # = FPS for 1s horizon
    OVERSAMPLE_RATE = 2
    SINGLE_HAND = False
    HAND_DETECTION_CONFIDENCE_THRESHOLD = 0.5
    # SUBSET = "small_subset"
    # SUBSET = "smaller_subset"
    SUBSET = None

    NEW_IMAGE_W = 640
    NEW_IMAGE_H = 480

    DEBUG = False   
    SAVE_ANNOTATED_IMAGES = True
    N_WORKERS = 16

    # desc = f"{STEP_SIZE}x{OVERSAMPLE_RATE}"

    # if SUBSET is not None:
    #     desc += f"_{SUBSET}"

    # if SINGLE_HAND:
    #     desc += "_singlehand"

    # if DEBUG:
    #     desc += "_debug"

    desc = ""
    BASE_S3_DOWNLOAD_DIR = "s3://tri-ml-datasets/cv_datasets/downloads/EgoDex"
    # BASE_LOCAL_DOWNLOAD_DIR = f"/home/ubuntu/fsx/HAMSTER_data/EgoDex/raw_{desc}"
    # BASE_LOCAL_PROCESSED_DIR = f"/home/ubuntu/fsx/HAMSTER_data/EgoDex/{desc}"
    BASE_S3_UPLOAD_DIR = "s3://robotics-manip-lbm/swatigupta/egomimic_data/egodex/processed/"

    BASE_LOCAL_DOWNLOAD_DIR = "datasets/egodex/raw"
    BASE_LOCAL_PROCESSED_DIR = "datasets/egodex/processed"

    # data_splits = ["part1"]
    data_splits = ["part2"]
    # data_splits = ["part3", "part4", "part5", "extra"]
    # 4 tasks to process for split: part1^M
    # ['add_remove_lid', 'clean_cups', 'clean_tableware', 'declutter_desk']
    # 2 tasks to process for split: part2
    # ['basic_fold', 'basic_pick_place']

    im_dims = np.array([NEW_IMAGE_W, NEW_IMAGE_H])

    # check aws sso login - skip for local processing
    try:
        boto3.client('s3').list_buckets()
    except ClientError as e:
        print (e)
        print("AWS SSO login required. Please run 'aws sso login' and try again.")
        exit(1)

    # current date-time string
    current_time = time.strftime("%Y%m%d-%H%M%S")
    print (f"Processing started at {current_time}")
    
    # Skip S3 task listing, use local directories directly
    task_list_path = BASE_LOCAL_DOWNLOAD_DIR + "/tasks_list_subset.txt"
    task_list = load_from_task_list(task_list_path)

    for data_split in data_splits:
        s3_split_download_dir = os.path.join(BASE_S3_DOWNLOAD_DIR, data_split)
        task_dirs = list_s3_subdirectories(s3_split_download_dir)
        s3_task_names = [task_dir.strip("/").split("/")[-1] for task_dir in task_dirs]

        tasks_subset = [task_name for task_name in s3_task_names if task_name in task_list]
        print (len(tasks_subset), "tasks to process for split:", data_split)
        print (tasks_subset)
        # tasks_subset.clear()
        # tasks_subset = ['clean_tableware']
        print (f"Processing {len(tasks_subset)} tasks for split: {data_split}")
        print ("Tasks:", tasks_subset)
        # print("\n" + "=" * 30 + f" Processing {data_split} " + "=" * 30)
        # print(f"task_names:", task_names)
        # print(f"len(task_names):", len(task_names))

        # if SUBSET is not None:
        #     task_names = [task_name for task_name in task_names if task_name in tasks_subset]
        #     print(f"[after subset filtering] task_names:", task_names)
        #     print(f"[after subset filtering] len(task_names):", len(task_names))

        # task_names = os.listdir(os.path.join(BASE_LOCAL_DOWNLOAD_DIR, data_split))

        datalist = []
        local_processed_dir = os.path.join(BASE_LOCAL_PROCESSED_DIR, data_split)
        for i, task in enumerate(tasks_subset):
            print ("Processing task: ", task)
            # if task == "clean_tableware":
            #     print ("Skipping task clean_tableware as it is already processed")
            #     continue
            hdf5_write_path = f"{local_processed_dir}/{task}.hdf5"

            # overwrite existing files
            # if os.path.exists(hdf5_write_path) and os.path.getsize(hdf5_write_path) > 1024 * 1024:
            #     print(f"Output HDF5 {hdf5_write_path} already exists, skipping task")
            #     continue

            # Skip S3 download, data already local
            local_task_download_dir = os.path.join(BASE_LOCAL_DOWNLOAD_DIR, data_split, task)
            s3_task_download_dir = os.path.join(s3_split_download_dir, task)
            if not os.path.exists(os.path.join(BASE_LOCAL_DOWNLOAD_DIR, data_split, task)):
                download_from_s3(s3_task_download_dir, local_task_download_dir)
            else:
                print (f"Local dir {local_task_download_dir} already exists, skipping S3 download")
            
            
            if not os.path.exists(local_task_download_dir):
                print(f"Error: Local task directory {local_task_download_dir} does not exist!")
                continue

            os.makedirs(local_processed_dir, exist_ok=True)

            # # Initialize empty HDF5 file
            with h5py.File(hdf5_write_path, "w") as f:
                f.create_group("data")
            
            num_successful = process_data_into_egodex_format(local_task_download_dir, 
                                            local_processed_dir,
                                            desc,
                                            N_WORKERS,
                                            STEP_SIZE,
                                            im_dims,
                                            SAVE_ANNOTATED_IMAGES,
                                            OVERSAMPLE_RATE,
                                            SINGLE_HAND,
                                            HAND_DETECTION_CONFIDENCE_THRESHOLD,
                                            hdf5_write_path,
                                            batch_size=N_WORKERS * 2)  # Process 2x worker count per batch
            
            if num_successful == 0:
                print(f"No episodes processed successfully for task {task}")
                # Remove empty HDF5 file
                if os.path.exists(hdf5_write_path):
                    os.remove(hdf5_write_path)
                if os.path.exists(local_task_download_dir):
                    shutil.rmtree(local_task_download_dir)
                continue

            split_train_val_from_hdf5(hdf5_path=hdf5_write_path, val_ratio=0.2)
            f.close()
            print (f"Saved {hdf5_write_path}")
            upload_to_s3_flag = False
            if upload_to_s3_flag is True:
                if upload_to_s3(hdf5_write_path, BASE_S3_UPLOAD_DIR):
                    print (f"Uploaded {hdf5_write_path} to {BASE_S3_UPLOAD_DIR} successfully")
                    # clean up local download dir to save space  
                    # os.remove(hdf5_write_path)  # Remove file, not directory
                    shutil.rmtree(local_task_download_dir)
            # exit(1)  # ### DEBUG ###


"""
AWS_REGION=us-east-1  python3 egomimic/process_egodex_to_egomimic.py 2>&1 | tee process_egodex_std_stderr2.txt

AWS_REGION=us-east-1 script -c "python3 egomimic/process_egodex_to_egomimic.py" process_egodex_$(date +%Y%m%d_%H%M%S).log

processed dir structure 
desc/
    data_split/
        dataset.json
        task/
            task_dataset.json
            images/
                episode_0/
                    frame_XXX.png 
                    ...
                    frame_XXX.png

AWS_REGION=us-east-1 aws s3 cp s3://robotics-manip-lbm/swatigupta/egomimic_data/egodex/processed/add_remove_lid.hdf5 .
"""