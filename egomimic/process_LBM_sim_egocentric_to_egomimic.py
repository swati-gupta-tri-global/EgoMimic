import os
import numpy as np
from tqdm import tqdm
import yaml
import ipdb
from egomimic.utils.egomimicUtils import (
    cam_frame_to_cam_pixels,
    ee_pose_to_cam_frame,
    interpolate_arr,
    REALSENSE_INTRINSICS
)
import ipdb

import h5py
import json
from egomimic.scripts.masking.utils import *
from panda_conversions import update_project_single_joint_position_to_image
# from robomimic.utils.s3_utils import download_s3_folder

OOD_TASKS = ["PutCupOnSaucer", "TurnCupUpsideDown", "TurnMugRightsideUp",
             "PutKiwiInCenterOfTable", "BimanualPutMugsOnPlatesFromTable", "BimanualPlaceAvocadoFromBowlOnCuttingBoard", "BimanualPlaceAppleFromBowlIntoBin", "BimanualLayCerealBoxOnCuttingBoardFromTopShelf"]
VAL_RATIO = 0.05
base_s3_episode_path = "s3://robotics-manip-lbm/kylehatch/video_cotrain/HAMSTER_data/LBM_sim_egocentric/raw/data/tasks/{}/{}/{}/bc/teleop/{}/diffusion_spartan/episode_{}/processed/"

def load_yaml(file_path):
    with open(file_path, "r") as file:
        data = yaml.safe_load(file)  # Use safe_load to prevent code execution
    return data

def inspect_npz_keys(data):
    for key in sorted(data.keys()):
        print (f"Key: {key}")
        try:
            shape = data[key].shape if hasattr(data[key], 'shape') else 'N/A'
            dtype = data[key].dtype if hasattr(data[key], 'dtype') else 'N/A'
            print(f"  {key}: shape={shape}, dtype={dtype}")
        except Exception as e:
            print(f"  {key}: Error retrieving shape/dtype - {e}")
    return list(data.keys())

def sam_processing(dataset, debug=False):
    """
    Applying masking to all images in the dataset

    dataset: path to the hdf5 file
    """
    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    sam = SAM()

    with h5py.File(dataset, "r+") as data:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for i in tqdm(range(len(data["data"].keys()))):
                demo = data[f"data/demo_{i}"]
                imgs = demo["obs/front_img_1"]
                ee_poses = demo["obs/ee_pose"]
                # extrinsics = demo["obs/extrinsics"]
                # intrinsics = demo["obs/intrinsics"]
                # depth = demo["obs/front_img_1_depth"]

                # import ipdb; ipdb.set_trace()
                # joint_positions = demo["obs/joint_positions"] # (n x 7)
                # gripper_pos_left, wrist_pos_left, elbow_pos_left = update_project_single_joint_position_to_image(
                #     joint_positions[:, :7],  # left arm
                #     extrinsics,
                #     intrinsics,
                #     arm="left",
                # )
                # gripper_pos_right, wrist_pos_right, elbow_pos_right = update_project_single_joint_position_to_image(
                #     joint_positions[:, 7:],  # right arm
                #     extrinsics,
                #     intrinsics,
                #     arm="right",
                # )
                print (ee_poses.shape)
                # ee_poses = np.concatenate([wrist_pos_left, wrist_pos_right], axis=1)  # gripper positions of left and right arms

                # Just average intrinsics across all frames
                # intrinsics = np.average(intrinsics, axis=0)

                overlayed_imgs, masked_imgs, raw_masks = sam.get_hand_mask_line_batched(imgs, ee_poses, intrinsics, depth, debug=debug)
                
                if "front_img_1_masked" in demo["obs"]:
                    print("Deleting existing masked images")
                    del demo["obs/front_img_1_masked"]
                if "front_img_1_line" in demo["obs"]:
                    print("Deleting existing line images")
                    del demo["obs/front_img_1_line"]
                if "front_img_1_mask" in demo["obs"]:
                    print("Deleting existing masks")
                    del demo["obs/front_img_1_mask"]

                demo["obs"].create_dataset("front_img_1_masked", data=masked_imgs, chunks=(1, 480, 640, 3))
                demo["obs"].create_dataset("front_img_1_mask", data=raw_masks, chunks=(1, 480, 640), dtype=bool)
                demo["obs"].create_dataset("front_img_1_line", data=overlayed_imgs, chunks=(1, 480, 640, 3))

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

def process_raw_data(inpt):
    episodes_list = [
        # "datasets/LBM_sim_egocentric/PutOrangeOnSaucer/cabot/sim/bc/teleop/2024-11-14T15-32-17-08-00/diffusion_spartan/episode_0/processed/",
        # "datasets/LBM_sim_egocentric/PutOrangeOnSaucer/cabot/sim/bc/teleop/2024-11-14T15-32-17-08-00/diffusion_spartan/episode_1/processed/",
        # "datasets/LBM_sim_egocentric/PutOrangeOnSaucer/cabot/sim/bc/teleop/2024-11-14T15-32-17-08-00/diffusion_spartan/episode_2/processed/"
        "datasets/LBM_sim_egocentric/data/tasks/BimanualHangMugsOnMugHolderFromDryingRack/riverway/sim/bc/teleop/2024-12-16T11-49-42-05-00/diffusion_spartan/episode_0/processed/",
        "datasets/LBM_sim_egocentric/data/tasks/BimanualHangMugsOnMugHolderFromDryingRack/riverway/sim/bc/teleop/2024-12-16T11-49-42-05-00/diffusion_spartan/episode_1/processed/",
        "datasets/LBM_sim_egocentric/data/tasks/BimanualHangMugsOnMugHolderFromDryingRack/riverway/sim/bc/teleop/2024-12-16T11-49-42-05-00/diffusion_spartan/episode_2/processed/"
    ]
    # download_folder = "datasets/LBM_sim_egocentric/PutOrangeOnSaucer/cabot/sim/bc/teleop/2024-11-14T15-32-17-08-00/diffusion_spartan/episode_2/processed/"
    # hdf5_path = "datasets/LBM_sim_egocentric/converted/PutOrangeOnSaucer.hdf5"
    hdf5_path = "datasets/LBM_sim_egocentric/converted/BimanualHangMugsOnMugHolderFromDryingRack.hdf5"
    print ("hdf5_path: ", hdf5_path)
    with h5py.File(hdf5_path, "w") as f:
        data = f.create_group("data")
        for ep_idx, episode in tqdm(enumerate(episodes_list), total=len(episodes_list), desc=f"process"):
            # station, task, episode_prefix = episode.split("/")

            # date, episode_number = episode_prefix.split("-episode-")

            # raw_episode_path = base_s3_episode_path.format(task, station, real_or_sim, date, int(episode_number))

            # assert f"{station}/{task}" == station_and_task_name, f"task: {task}, station_and_task_name: {station_and_task_name}, raw_episode_path: {raw_episode_path}"

            # download_folder = os.path.join(tmp_folder_base, f"process_{process_no}")
            # os.makedirs(download_folder, exist_ok=True)
            # download_s3_folder(raw_episode_path, download_folder)
            download_folder = episode
            print ("Download folder:", download_folder)
            task = "PutOrangeOnSaucer"

            language_instructions = load_yaml("~/Egomimic/egomimic/language_annotations.yaml")["language_dict"]
            possible_task_descriptions = language_instructions[task]["original"] + language_instructions[task]["randomized"]

            meta_data_file = os.path.join(download_folder, "metadata.yaml")
            if not os.path.isfile(meta_data_file):
                print(f"{meta_data_file} does not exist") 
            else:
                meta_data = load_yaml(meta_data_file)
                observations_file = os.path.join(download_folder, "observations.npz")
                observations = np.load(observations_file)

                # inspect_npz_keys(observations)
                camera_names = {val:key for key, val in meta_data["camera_id_to_semantic_name"].items()}

                for camera_name in ["scene_right_0"]:
                    camera_id = camera_names[camera_name]
                    front_img_1 = observations[camera_id]
                    front_img_1_depth = observations[f"{camera_id}_depth"]
                    intrinsics = np.load(os.path.join(download_folder, "intrinsics.npz"))[camera_id]
                    print (intrinsics)
                    extrinsics = np.load(os.path.join(download_folder, "extrinsics.npz"))[camera_id][0]
                    print (f"images shape: {front_img_1.shape}, intrinsics shape: {intrinsics.shape}, extrinsics shape: {extrinsics.shape}")

                actions_file = os.path.join(download_folder, "actions.npz")
                actions = np.load(actions_file, allow_pickle=True)["actions"] # shape=(124, 20) --> ac_dim=20 (bimanual)


                pose_xyz_left = observations["robot__actual__poses__right::panda__xyz"] # shape=(124, 3) --> obs/ee_pose (left hand_pose)
                pose_xyz_right = observations["robot__actual__poses__left::panda__xyz"] # shape=(124, 3) --> obs/ee_pose (right hand_pose)
                # import ipdb; ipdb.set_trace()
                pose_xyz_left = ee_pose_to_cam_frame(pose_xyz_left, extrinsics)[:, :3]
                pose_xyz_right = ee_pose_to_cam_frame(pose_xyz_right, extrinsics)[:, :3]

                print (f"pose_xyz_left.shape: {pose_xyz_left.shape}, pose_xyz_right.shape: {pose_xyz_right.shape}")
                # gripper_state_left = observations["robot__actual__grippers__right::panda_hand"] # wether the gripper is open or closed
                # gripper_state_right = observations["robot__actual__grippers__left::panda_hand"]
                robot_joint_positions_left = observations["robot__actual__joint_position__left::panda"] # shape=(124, 7) --> obs/joint_positions
                robot_joint_positions_right = observations["robot__actual__joint_position__right::panda"] # shape=(124, 7) --> obs/joint_positions

                # import ipdb; ipdb.set_trace()
                ee_pose = np.hstack([pose_xyz_left, pose_xyz_right])
                joint_positions = np.hstack([robot_joint_positions_left, robot_joint_positions_right])
                print (f"ee_pose.shape: {ee_pose.shape}, joint_positions.shape: {joint_positions.shape}")
                ac_dim = actions.shape[1]

            
                data.attrs["env_args"] = json.dumps({})
                lbm_fps = 10
                horizon_seconds = 4.0
                N = actions.shape[0]
                print(f"{N} frames in file")
                chunk_size = int(N / horizon_seconds)  # Define chunk size
                print("chunk_size: ", chunk_size)
                group = data.create_group(f"demo_{ep_idx}")
                ac_reshape_interp = []
                for i in range(0, N):
                    if i + chunk_size > N:
                        print(f"Not enough data to create another chunk of size {chunk_size} at index {i}")
                        # copy-padding for the last few frames
                        ac_reshape = np.zeros((1, chunk_size, ac_dim))
                        ac_reshape[:, :N - i] = actions[i : N].reshape(1, -1, ac_dim)
                        ac_reshape[:, N - i :] = np.tile(
                            actions[N - 1].reshape(1, 1, ac_dim), (1, chunk_size - (N - i), 1)
                        )
                    else:
                        ac_reshape = actions[i : i + chunk_size].reshape(1, chunk_size, ac_dim)
                    
                    ac_reshape_interp.append(interpolate_arr(ac_reshape, 100))
                # import ipdb; ipdb.set_trace()
                ac_reshape_interp = np.concatenate(ac_reshape_interp, axis=0)
                # Ensure proper data type and finite values
                ac_reshape_interp = ac_reshape_interp.astype(np.float32)
                ac_reshape_interp = np.nan_to_num(ac_reshape_interp, nan=0.0, posinf=0.0, neginf=0.0)

                print (f"ac_reshape_interp.shape: {ac_reshape_interp.shape}")
                left_joint_act = ac_reshape_interp[:, :, :7]
                left_xyz_act = ac_reshape_interp[:, :, 7:10]
                right_joint_act = ac_reshape_interp[:, :, 10:17]   
                right_xyz_act = ac_reshape_interp[:, :, 17:20]

                combined_joint_act = np.concatenate([left_joint_act, right_joint_act], axis=2)
                combined_xyz_act = np.concatenate([left_xyz_act, right_xyz_act], axis=2)
                # import ipdb; ipdb.set_trace()

                print (f"combined_joint_act.shape: {combined_joint_act.shape}, combined_xyz_act.shape: {combined_xyz_act.shape}")

                # import ipdb; ipdb.set_trace()

                # print ("ac_reshape_interp.shape: ", ac_reshape_interp.shape)
                # print ("combined_joint_act.shape: ", combined_joint_act.shape)
                # print ("combined_xyz_act.shape: ", combined_xyz_act.shape)
                
                group.create_dataset("actions_joints_act", data=combined_joint_act)
                group.create_dataset("actions_xyz_act", data=combined_xyz_act)
                group.attrs["num_samples"] = int(ac_reshape_interp.shape[0])
                group.create_dataset(
                    "obs/front_img_1", data=front_img_1
                )
                group.create_dataset(
                    "obs/front_img_1_depth", data=front_img_1_depth
                )
                group.create_dataset("obs/ee_pose", data=ee_pose)
                group.create_dataset("obs/joint_positions", data=joint_positions)
                # intrinsics_3x4 = np.hstack([intrinsics[:, :3], 
                #            np.array([[0], [0], [0]])])
                # group.create_dataset(
                #     "obs/intrinsics", data=np.tile(intrinsics_3x4[None, :], (N, 1, 1))
                # )
                # group.create_dataset(
                #     "obs/extrinsics", data=np.tile(extrinsics[None, :], (N, 1, 1))
                # )

    split_train_val_from_hdf5(hdf5_path=hdf5_path, val_ratio=VAL_RATIO)

    # import ipdb; ipdb.set_trace()
    # sam_processing(hdf5_path, debug=True)
    # import ipdb; ipdb.set_trace()

process_raw_data("input_placeholder")  # Replace with actual input if needed

# Train command: swatigupta@Puget-248656:~/EgoMimic/egomimic$
# python scripts/pl_train.py --config configs/egomimic_oboo.json --dataset ../datasets/LBM_sim_egocentric/converted/BimanualHangMugsOnMugHolderFromDryingRack.hdf5 --debug