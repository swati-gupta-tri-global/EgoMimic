import os
import numpy as np
from glob import glob
from tqdm import tqdm, trange
import time
import cv2 
import imageio
from collections import defaultdict
from torchcodec.decoders import VideoDecoder
import json 
import yaml 
import h5py
import boto3
from botocore.exceptions import ClientError
import shutil 
import subprocess
from urllib.parse import urlparse
from multiprocessing import Pool
from rdp import rdp
import random
from egomimic.utils.egomimicUtils import (
    ee_pose_to_cam_frame,
    interpolate_arr,
)

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
    command = ["s5cmd", "cp", s3_path, local_path]
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

def upload_to_s3(local_path, s3_path, s3_command="sync", extra_options=None):
    command = ["aws", "s3", s3_command, local_path, s3_path]

    if extra_options is not None:
        command += extra_options

    print(f"Uploading {local_path} to {s3_path}...")
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    # process = subprocess.Popen(command)
    stdout, stderr = process.communicate()

    if process.returncode == 0:
        # print("Download successful!")
        return True
    else:
        print(f"Error uploading folder:\n{stderr}")
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
    
    with h5py.File(hdf5_file, 'r') as f:
        # f.keys(): <KeysViewHDF5 ['camera', 'confidences', 'transforms']>
        camera_extrinsics =  f["transforms"]["camera"][:]  # (300, 4, 4)
        left_idxfinger_knuckle_3D_poses = f["transforms"]["leftHand"][:]  # (300, 4, 4)
        right_idxfinger_knuckle_3D_poses = f["transforms"]["rightHand"][:]

        if "confidences" in f:
            left_idxfinger_knuckle_confidences = f["confidences"]["leftHand"][:] # (300, )
            right_idxfinger_knuckle_confidences = f["confidences"]["rightHand"][:]

        if left_idxfinger_knuckle_confidences.max() < hand_detection_confidence_threshold or right_idxfinger_knuckle_confidences.max() < hand_detection_confidence_threshold:
            print (f"[Warning] all hand tracks detected below confidence threshold {hand_detection_confidence_threshold} for episode {ep_no}, skipping episode")
            # skip
            return episode_data_list

        N = camera_extrinsics.shape[0]
        
    
        decoder = VideoDecoder(mp4_file, device="cpu")
        assert decoder.metadata.num_frames == N, f"decoder.metadata.num_frames: {decoder.metadata.num_frames}, N: {N}"
        # print ("Seq len: ", N, step_size, decoder.metadata.num_frames)
        # original_im_dims = np.array([decoder.metadata.width, decoder.metadata.height])
   
        images = []
        ee_poses = []
        actions_xyz = []
        actions_xyz_act = []  # interpolated actions

        
        for i in trange(0, decoder.metadata.num_frames, desc=f"[Processing episode {ep_no}]"):
            initial_image = np.moveaxis(decoder[i].cpu().detach().numpy(), 0, -1)
            initial_image = cv2.resize(initial_image, im_dims)
            
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
                print (f"[Partial valid] {len(valid_indices)}/{step_size} hand tracks detected above confidence threshold {hand_detection_confidence_threshold} for episode {ep_no}, step starting at {i}, skipping this chunk")
                continue

            # Our goal is to construct the action-chunk (a_t -> a_t+H) by transforming each position in the trajectory into the
            # observation camera frame Ft
            left_hand_actions = ee_pose_to_cam_frame(path_segments["left"]["3D_poses"][valid_indices, 3, :3], camera_extrinsics[i])  # (num_valid, 3)
            right_hand_actions = ee_pose_to_cam_frame(path_segments["right"]["3D_poses"][valid_indices, 3, :3], camera_extrinsics[i])
            hand_actions = np.hstack((left_hand_actions, right_hand_actions))  # (num_valid, 6)
            
            actions_xyz.append(hand_actions)
            images.append(initial_image)
            # Convert EE pose also to camera frame Ft
            left_ee_pose = ee_pose_to_cam_frame(left_idxfinger_knuckle_3D_poses[i][3, :3].reshape(1, -1), camera_extrinsics[i])
            right_ee_pose = ee_pose_to_cam_frame(right_idxfinger_knuckle_3D_poses[i][3, :3].reshape(1, -1), camera_extrinsics[i])
            stacked_ee_pose = np.hstack((left_ee_pose, right_ee_pose))  # at current time step # (6, )
            ee_poses.append(stacked_ee_pose)

        
        if len(images) == 0:
            print (f"[Skipping] No valid data for episode {ep_no}")
            return episode_data_list
        
        # print (f"len(images): {len(images)}, len(ee_poses): {len(ee_poses)}, len(actions_xyz): {len(actions_xyz)}, actions_xyz[0].shape: {actions_xyz[0].shape}, actions_xyz[-1].shape: {actions_xyz[-1].shape}")

        actions_xyz = np.array(actions_xyz)
        actions_xyz_act = interpolate_arr(actions_xyz, 100)  # (num_valid * oversample_rate, 6)
        episode_data_list = dict(ep_no=ep_no, 
                    front_img_1=np.array(images), 
                    actions_xyz=actions_xyz,
                    actions_xyz_act=actions_xyz_act,
                    ee_pose=np.squeeze(np.array(ee_poses)))


    return episode_data_list

def process_data_into_egodex_format(local_download_dir, 
                                     local_processed_dir, 
                                     desc,
                                     n_workers, 
                                     step_size, 
                                     im_dims,
                                     save_annotated_images, 
                                     oversample_rate, 
                                     singlehand,
                                     hand_detection_confidence_threshold):

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

    function_inpts = []
    for ep_no in episode_numbers:
        mp4_file = os.path.join(local_download_dir, f"{ep_no}.mp4")
        hdf5_file = os.path.join(local_download_dir, f"{ep_no}.hdf5")
        function_inpts.append((ep_no, mp4_file, hdf5_file, step_size, oversample_rate, hand_detection_confidence_threshold, im_dims, save_annotated_images))
        # if ep_no > 2:
        #     break  # ### DEBUG ###

    # episode_data_dicts = [process_single_episode(inpt) for inpt in function_inpts] ### DEBUG ###
    with Pool(n_workers) as p:
        episode_data_dicts = list(p.imap(process_single_episode, function_inpts))
    
    # import ipdb; ipdb.set_trace()
    task_datadict = {}
    for episode_data_dict in episode_data_dicts:
        if len(episode_data_dict) == 0:
            print ("[Skipping] No valid data for episode")
            continue
        task_datadict[episode_data_dict["ep_no"]] = episode_data_dict

    return task_datadict

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
    SAVE_ANNOTATED_IMAGES = False
    N_WORKERS = 16

    # desc = f"{STEP_SIZE}x{OVERSAMPLE_RATE}"

    # if SUBSET is not None:
    #     desc += f"_{SUBSET}"

    # if SINGLE_HAND:
    #     desc += "_singlehand"

    # if DEBUG:
    #     desc += "_debug"

    desc = ""
    BASE_S3_DOWNLOAD_DIR = f"s3://tri-ml-datasets/cv_datasets/downloads/EgoDex"
    # BASE_LOCAL_DOWNLOAD_DIR = f"/home/ubuntu/fsx/HAMSTER_data/EgoDex/raw_{desc}"
    # BASE_LOCAL_PROCESSED_DIR = f"/home/ubuntu/fsx/HAMSTER_data/EgoDex/{desc}"
    # BASE_S3_UPLOAD_DIR = f"s3://robotics-manip-lbm/kylehatch/video_cotrain/HAMSTER_data/EgoDex/{desc}"

    BASE_LOCAL_DOWNLOAD_DIR = f"/home/swatigupta/EgoMimic/datasets/egodex/raw"
    BASE_LOCAL_PROCESSED_DIR = f"/home/swatigupta/EgoMimic/datasets/egodex/processed"

    # data_splits = ["test", "part1", "part2", "part3", "part4", "part5", "extra"]
    data_splits = ["part1", "part2", "part3", "part4", "part5", "extra"]
    # data_splits = ["test"]

    im_dims = np.array([NEW_IMAGE_W, NEW_IMAGE_H])
    
    task_list_path = BASE_LOCAL_DOWNLOAD_DIR + "/tasks_list_subset.txt"
    task_list = load_from_task_list(task_list_path)

    for data_split in data_splits:
        s3_split_download_dir = os.path.join(BASE_S3_DOWNLOAD_DIR, data_split)
        task_dirs = list_s3_subdirectories(s3_split_download_dir)
        s3_task_names = [task_dir.strip("/").split("/")[-1] for task_dir in task_dirs]

        tasks_subset = [task_name for task_name in s3_task_names if task_name in task_list]
        print (len(tasks_subset), "tasks to process for split:", data_split)
        print (tasks_subset)
        # print("\n" + "=" * 30 + f" Processing {data_split} " + "=" * 30)
        # print(f"task_names:", task_names)
        # print(f"len(task_names):", len(task_names))

        # if SUBSET is not None:
        #     task_names = [task_name for task_name in task_names if task_name in tasks_subset]
        #     print(f"[after subset filtering] task_names:", task_names)
        #     print(f"[after subset filtering] len(task_names):", len(task_names))

        # task_names = os.listdir(os.path.join(BASE_LOCAL_DOWNLOAD_DIR, data_split))

        datalist = []
        for i, task in enumerate(tasks_subset):
            print ("Processing task: ", task)

            hdf5_write_path = f"{local_processed_dir}/{task}.hdf5"
            if os.path.exists(hdf5_write_path) and os.path.getsize(hdf5_write_path) > 0:
                print(f"Output HDF5 {hdf5_write_path} already exists, skipping task")
                continue

            s3_task_download_dir = os.path.join(s3_split_download_dir, task)
            if not os.path.exists(os.path.join(BASE_LOCAL_DOWNLOAD_DIR, data_split, task)):
                download_from_s3(s3_task_download_dir, os.path.join(BASE_LOCAL_DOWNLOAD_DIR, data_split, task))
            else:
                print (f"Local dir {os.path.join(BASE_LOCAL_DOWNLOAD_DIR, data_split, task)} already exists, skipping S3 download")
            # s3_download_dir = os.path.join(s3_split_download_dir, task)
            local_download_dir = os.path.join(BASE_LOCAL_DOWNLOAD_DIR, data_split, task)
            local_processed_dir = os.path.join(BASE_LOCAL_PROCESSED_DIR, data_split)
            # s3_upload_dir = os.path.join(BASE_S3_UPLOAD_DIR, data_split, task)

            os.makedirs(local_processed_dir, exist_ok=True)

            # task_dataset_json_s3_uri = os.path.join(s3_upload_dir, "task_dataset.json")
            # task_dataset_json_file = os.path.join(local_processed_dir, "task_dataset.json")

            task_datalist = process_data_into_egodex_format(local_download_dir, 
                                            local_processed_dir,
                                            desc,
                                            N_WORKERS,
                                            STEP_SIZE,
                                            im_dims,
                                            SAVE_ANNOTATED_IMAGES,
                                            OVERSAMPLE_RATE,
                                            SINGLE_HAND,
                                            HAND_DETECTION_CONFIDENCE_THRESHOLD)

            # Save an hdf5 per task
            hdf5_write_path = f"{local_processed_dir}/{task}.hdf5"
            with h5py.File(hdf5_write_path, "w") as f:
                    data = f.create_group("data")
                    for idx, (ep_idx, episode_data) in enumerate(task_datalist.items()):
                        if len(episode_data) == 0:
                            print (f"[Skipping] No valid data for episode {ep_idx} of task {task}")
                            continue
                        group = data.create_group(f"demo_{idx}")
                        group.create_dataset("actions_xyz", data=episode_data["actions_xyz"])
                        group.create_dataset("actions_xyz_act", data=episode_data["actions_xyz_act"])
                        group.create_dataset(
                            "obs/front_img_1", data=episode_data["front_img_1"]
                        )
                        group.create_dataset("obs/ee_pose", data=episode_data["ee_pose"])
                        group.attrs["num_samples"] = int(episode_data["actions_xyz"].shape[0])

            split_train_val_from_hdf5(hdf5_path=hdf5_write_path, val_ratio=0.2)
            print (f"Saved {hdf5_write_path}")

            # clean up local download dir to save space
            if os.path.exists(local_download_dir):
                shutil.rmtree(local_download_dir)
            # exit(1)  # ### DEBUG ###


"""
python3 egomimic/process_egodex_data.py 2>&1 | tee process_egodex_std_stderr2.txt

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

"""