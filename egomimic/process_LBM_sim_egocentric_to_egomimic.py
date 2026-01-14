import os
import numpy as np
from tqdm import tqdm
import yaml
import ipdb
import pandas as pd
import argparse
from concurrent.futures import ThreadPoolExecutor
from egomimic.utils.egomimicUtils import (
    # cam_frame_to_cam_pixels,
    ee_pose_to_cam_frame,
    interpolate_arr
)
import subprocess
import shutil
import h5py
import json
# from egomimic.scripts.masking.utils import *
# export PATH=$PATH:~/go/bin for s5cmd
# python egomimic/process_LBM_sim_egocentric_to_egomimic.py --task_filter BimanualHangMugsOnMugHolderFromTable --csv_path filtered_output.csv --download_from_s3

"""
root@ee3cde590bca:/workspace/externals/EgoMimic# h5ls -r  datasets/LBM_sim_egocentric/processed/TurnMugRightsideUp.hdf5 | grep demo_0/
/data/demo_0/actions_joints_act Dataset {321, 100, 16}
/data/demo_0/actions_xyz_act Dataset {321, 100, 6}
/data/demo_0/obs         Group
/data/demo_0/obs/ee_pose Dataset {321, 6}
/data/demo_0/obs/extrinsics Dataset {321, 4, 4}
/data/demo_0/obs/front_img_1 Dataset {321, 480, 640, 3}
/data/demo_0/obs/intrinsics Dataset {321, 3, 3}
/data/demo_0/obs/joint_positions Dataset {321, 14}
"""
OOD_TASKS = ["PutCupOnSaucer", "TurnCupUpsideDown", "TurnMugRightsideUp",
             "PutKiwiInCenterOfTable", "BimanualPutMugsOnPlatesFromTable", "BimanualPlaceAvocadoFromBowlOnCuttingBoard", "BimanualPlaceAppleFromBowlIntoBin", "BimanualLayCerealBoxOnCuttingBoardFromTopShelf"]
VAL_RATIO = 0.05
base_s3_episode_path = "s3://robotics-manip-lbm/kylehatch/video_cotrain/HAMSTER_data/LBM_sim_egocentric/raw/data/tasks/{}/{}/{}/bc/teleop/{}/diffusion_spartan/episode_{}/processed/"

def load_yaml(file_path):
    with open(file_path, "r") as file:
        data = yaml.safe_load(file)  # Use safe_load to prevent code execution
    return data

def download_s3_folder(s3_path, local_folder):
    command = ["aws", "s3", "sync", s3_path, local_folder]
    # command = ["aws", "s3", "sync", s3_path, local_folder, "--exclude", "*", "--include", "observations.npz"]

    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate()

        if process.returncode == 0:
            # print("Download successful!")
            return True
        else:
            print(f"Error downloading folder:\n{stderr}")
            return False
    except Exception as e:
        print(f"Exception occurred: {e}")
        return False

def download_s3_folder_s5cmd(s3_path, local_folder):
    command = ["s5cmd", "cp", s3_path, local_folder]
    print ("s5cmd command: ", command)
    # s5cmd cp  s3://robotics-manip-lbm/kylehatch/video_cotrain/HAMSTER_data/LBM_sim_egocentric/raw/data/tasks/BimanualPlacePearFromBowlOnCuttingBoard/riverway/sim/bc/teleop/*/diffusion_spartan/episode_*/processed/ datasets/LBM_sim_egocentric/data/tasks/BimanualPlacePearFromBowlOnCuttingBoard/riverway/sim/bc/teleop
    # command = ["aws", "s3", "sync", s3_path, local_folder, "--exclude", "*", "--include", "observations.npz"]

    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate()

        if process.returncode == 0:
            # print("Download successful!")
            return True
        else:
            print(f"Error downloading folder:\n{stderr}")
            return False
    except Exception as e:
        print(f"Exception occurred: {e}")
        return False
    

def download_task_data_from_s3(task_name, environment, sim_or_real, base_s3_path, local_base_path):
    """
    Download task data from S3 for a specific task and environment
    Returns list of local episode paths
    """
    local_task_path = os.path.join(local_base_path, task_name, environment, sim_or_real, "bc", "teleop")
    os.makedirs(local_task_path, exist_ok=True)
    
    # Try to find episodes by looking for common timestamp patterns
    # This is a simplified approach - in practice, you might need to list S3 contents
    base_s3_episode_path_ = "s3://robotics-manip-lbm/kylehatch/video_cotrain/HAMSTER_data/LBM_sim_egocentric/raw/data/tasks/{}/{}/{}/bc/teleop/{}/diffusion_spartan/episode_{}/processed/{}"
    # s3://robotics-manip-lbm/kylehatch/video_cotrain/HAMSTER_data/LBM_sim_egocentric/raw/data/tasks/BimanualPlacePearFromBowlOnCuttingBoard/riverway/sim/bc/teleop/2025-01-06T13-51-34-08-00/diffusion_spartan/episode_0
    # s5cmd cp "s3://robotics-manip-lbm/kylehatch/video_cotrain/HAMSTER_data/LBM_sim_egocentric/raw/data/tasks/BimanualPlacePearFromBowlOnCuttingBoard/riverway/sim/bc/teleop/*/diffusion_spartan/*/processed/*" datasets/LBM_sim_egocentric/data/tasks/BimanualPlacePearFromBowlOnCuttingBoard/riverway/sim/bc/teleop/
    success = download_s3_folder_s5cmd(base_s3_episode_path_.format(task_name, environment, sim_or_real, "*", "*", "*"), local_task_path)

    if success:
        print(f"Successfully downloaded data for {task_name} in {environment} ({sim_or_real})")
    else:
        print(f"Failed to download data for {task_name} in {environment} ({sim_or_real})")

def load_tasks_from_csv(csv_path):
    """
    Load tasks from the filtered_output.csv file
    Returns a list of task names that have data available
    """
    df = pd.read_csv(csv_path)
    # Filter tasks that have at least some data (total > 0)
    tasks_with_data = df[df['total'] > 0]['Task'].tolist()
    return tasks_with_data

def get_task_environments(csv_path, task_name, real_flag=True):
    """
    Get available environments for a specific task from the CSV
    Returns a list of (environment, count) tuples
    """
    df = pd.read_csv(csv_path)
    task_row = df[df['Task'] == task_name].iloc[0]
    
    environments = []
    # Check simulation environments
    sim_envs = ['sim_cabot', 'sim_riverway']
    for env in sim_envs:
        if task_row[env] > 0:
            environments.append((env.replace('sim_', ''), 'sim', task_row[env]))
    
    # Check real environments
    if real_flag:
        real_envs = ['real_wood_island', 'real_hersey', 'real_maverick', 'real_riverway', 
                    'real_ruggles', 'real_salem', 'real_davis', 'real_milton', 'real_wollaston']
        for env in real_envs:
            if task_row[env] > 0:
                environments.append((env.replace('real_', ''), 'real', task_row[env]))
    
    return environments

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

def process_episode_parallel(episode_data):
    """Process a single episode in parallel"""
    ep_idx, episode_path = episode_data
    
    try:
        # print(f"Processing episode {ep_idx}: {episode_path}")

        meta_data_file = os.path.join(episode_path, "metadata.yaml")
        if not os.path.isfile(meta_data_file):
            print(f"{meta_data_file} does not exist, skipping episode")
            return None
            
        meta_data = load_yaml(meta_data_file)
        observations_file = os.path.join(episode_path, "observations.npz")
        observations = np.load(observations_file)

        # Get camera data
        camera_names = {val:key for key, val in 
                       meta_data["camera_id_to_semantic_name"].items()}

        for camera_name in ["scene_right_0"]:
            camera_id = camera_names[camera_name]
            front_img_1 = observations[camera_id]
            intrinsics = np.load(os.path.join(episode_path, 
                                            "intrinsics.npz"))[camera_id]
            extrinsics = np.load(os.path.join(episode_path, 
                                            "extrinsics.npz"))[camera_id][0]
            # print(f"Images shape: {front_img_1.shape}, "
            #       f"intrinsics shape: {intrinsics.shape}, "
            #       f"extrinsics shape: {extrinsics.shape}")

        # Load actions (Note that left and right convention seems to be flipped, see outputs of visualize_LBM_episode.py)
        # actions_file = os.path.join(episode_path, "actions.npz")
        # actions = np.load(actions_file, allow_pickle=True)["actions"]
        # print (f"actions:", actions[0])
        # Process poses
        pose_xyz_right = observations["robot__actual__poses__left::panda__xyz"]
        pose_xyz_left = observations["robot__actual__poses__right::panda__xyz"]
        pose_xyz_left = ee_pose_to_cam_frame(pose_xyz_left, extrinsics)[:, :3]
        pose_xyz_right = ee_pose_to_cam_frame(pose_xyz_right, extrinsics)[:, :3]

        # print(f"Pose shapes - left: {pose_xyz_left.shape}, "
        #       f"right: {pose_xyz_right.shape}")
        
    #     ipdb> print("Left joint ranges:",
    #   f"min={robot_joint_positions_left.min():.3f}, max={robot_joint_positions_left.max():.3f}")
    #     Left joint ranges: min=-2.427, max=2.645
    #     Right joint ranges: min=-2.356, max=2.561
        # Get joint positions
        robot_joint_positions_right = observations["robot__actual__joint_position__left::panda"] # (219, 7)
        robot_joint_positions_left = observations["robot__actual__joint_position__right::panda"]

        # check keys and shapes, use instead of action array if possible
        robot_position_action_right = observations["robot__desired__poses__left::panda__xyz"] # (219, 3)
        robot_position_action_left = observations["robot__desired__poses__right::panda__xyz"]
        robot_joint_action_right = observations["robot__desired__joint_position__left::panda"] # (219, 7)
        robot_joint_action_left = observations["robot__desired__joint_position__right::panda"]
        # robot__desired__poses__left::panda__rot_6d # (219, 6)
        robot_gripper_action_right = observations["robot__desired__grippers__left::panda_hand"] # (219, 1)
        robot_gripper_action_left = observations["robot__desired__grippers__right::panda_hand"]

        # Transform position actions to camera frame (same as poses)
        robot_position_action_left = ee_pose_to_cam_frame(robot_position_action_left, extrinsics)[:, :3]
        robot_position_action_right = ee_pose_to_cam_frame(robot_position_action_right, extrinsics)[:, :3]

        ee_pose = np.hstack([pose_xyz_left, pose_xyz_right])
        joint_positions = np.hstack([robot_joint_positions_left, 
                                   robot_joint_positions_right])

        # print(f"Actions shape: {actions.shape}")
        position_actions = np.hstack([robot_position_action_left, robot_position_action_right])
        joint_actions = np.hstack([robot_joint_action_left, robot_joint_action_right])
        gripper_actions = np.hstack([robot_gripper_action_left, robot_gripper_action_right])
        actions = np.hstack([position_actions, joint_actions, gripper_actions])
        # print(f"Joint actions shape: {joint_actions.shape}, "
        #       f"Position actions shape: {position_actions.shape}, "
        #       f"Gripper actions shape: {gripper_actions.shape}, "
        #       f"Combined actions shape: {actions.shape}")
        ac_dim = actions.shape[1]
        
        # Process actions with chunking
        horizon_seconds = 4.0
        N = joint_actions.shape[0]
        # print(f"{N} frames in episode")
        chunk_size = int(N / horizon_seconds)
        # print(f"Chunk size: {chunk_size}")
        
        ac_reshape_interp = []
            
        for i in range(0, N):
            if i + chunk_size > N:
                # print(f"Not enough data to create another chunk of size "
                #       f"{chunk_size} at index {i}, tiling last action")
                ac_reshape = np.zeros((1, chunk_size, ac_dim))
                ac_reshape[:, :N - i] = actions[i : N].reshape(1, -1, ac_dim)
                ac_reshape[:, N - i :] = np.tile(
                    actions[N - 1].reshape(1, 1, ac_dim), 
                    (1, chunk_size - (N - i), 1)
                )
            else:
                ac_reshape = actions[i : i + chunk_size].reshape(1, 
                                                               chunk_size, 
                                                               ac_dim)
            
            ac_reshape_interp.append(interpolate_arr(ac_reshape, 100))
        
        ac_reshape_interp = np.concatenate(ac_reshape_interp, axis=0)
        ac_reshape_interp = ac_reshape_interp.astype(np.float32)
        ac_reshape_interp = np.nan_to_num(ac_reshape_interp, nan=0.0, 
                                        posinf=0.0, neginf=0.0)

        # print(f"Action interpolation shape: {ac_reshape_interp.shape}")
        # Example action structure from LBM
        # action[ 0: 3] = xyz0
        # action[ 3: 9] = rot0
        # action[18:19] = grip0

        # action[ 9:12] = xyz1
        # action[12:18] = rot1
        # action[19:20] = grip1
        # left_xyz_act = ac_reshape_interp[:, :, :3]
        # left_joint_act = ac_reshape_interp[:, :, 3:9]  # BUG: this isnt joint angles, but rotation matrix entries for ee orientation
        # left_gripper_act = ac_reshape_interp[:, :, 18:19]
        # right_xyz_act = ac_reshape_interp[:, :, 9:12]
        # right_joint_act = ac_reshape_interp[:, :, 12:18]
        # right_gripper_act = ac_reshape_interp[:, :, 19:20]

        left_xyz_act = ac_reshape_interp[:, :, :3]
        right_xyz_act = ac_reshape_interp[:, :, 3:6]
        left_joint_act = ac_reshape_interp[:, :, 6:13]
        right_joint_act = ac_reshape_interp[:, :, 13:20]
        left_gripper_act = ac_reshape_interp[:, :, 20:21]
        right_gripper_act = ac_reshape_interp[:, :, 21:22]

        # print ("left gripper act val:", left_gripper_act[0, 0, 0:10], left_gripper_act[0, 1, 0:10])
        # print ("right gripper act val:", right_gripper_act[0, 0, 0:10], right_gripper_act[0, 1, 0:10])

        combined_joint_act = np.concatenate([left_joint_act, left_gripper_act, right_joint_act, right_gripper_act], 
                                          axis=2)
        combined_xyz_act = np.concatenate([left_xyz_act, right_xyz_act], axis=2)

        # print(f"Combined actions - joints: {combined_joint_act.shape}, "
        #       f"xyz: {combined_xyz_act.shape}")
        
        # Get sequence length from the actions
        seq_len = ac_reshape_interp.shape[0]
        
        # Repeat intrinsics and extrinsics for each timestep in the sequence
        # This ensures they have the same first dimension as other observations
        intrinsics_seq = np.tile(intrinsics[np.newaxis, :, :], (seq_len, 1, 1))  # (seq_len, 3, 3)
        extrinsics_seq = np.tile(extrinsics[np.newaxis, :, :], (seq_len, 1, 1))  # (seq_len, 4, 4)
        
        # Return processed episode data
        return {
            'ep_idx': ep_idx,
            'combined_joint_act': combined_joint_act,
            'combined_xyz_act': combined_xyz_act,
            'num_samples': int(ac_reshape_interp.shape[0]),
            'front_img_1': front_img_1,
            'ee_pose': ee_pose,
            'intrinsics': intrinsics_seq,
            'extrinsics': extrinsics_seq,
            'joint_positions': joint_positions
        }
        
    except Exception as e:
        print(f"Error processing episode {ep_idx} at {episode_path}: {e}")
        return None

def process_raw_data(csv_path, base_s3_path, local_base_path, output_base_path, download_from_s3=True, cleanup_local_data=True, max_workers=None):
    """
    Process raw data for all tasks specified in the CSV file
    
    Args:
        csv_path: Path to the filtered_output.csv file
        base_s3_path: Base S3 path template for downloading data
        local_base_path: Local base path for storing downloaded data
        output_base_path: Base path for output HDF5 files
        download_from_s3: Whether to download data from S3 or use existing local data
        cleanup_local_data: Whether to delete local data after processing
    """
    # Load tasks from CSV
    tasks = load_tasks_from_csv(csv_path)
    print(f"Found {len(tasks)} tasks with data: {tasks}")
    
    for task_name in tasks:
        print(f"\nProcessing task: {task_name}")
        
        # Get available environments for this task
        environments = get_task_environments(csv_path, task_name, real_flag=False)
        print(f"Available environments: {environments}")
        
        # Create output directory
        output_dir = os.path.join(output_base_path, "processed")
        os.makedirs(output_dir, exist_ok=True)
        
        output_hdf5_path = os.path.join(output_dir, f"{task_name}.hdf5")
        # check if size is greater than 1MB
        # if os.path.exists(output_hdf5_path) and os.path.getsize(output_hdf5_path) > 1024 * 1024:  # > 1MB
        #     print(f"Output HDF5 {output_hdf5_path} already exists, skipping task")
        #     continue

        f = h5py.File(output_hdf5_path, "w")
        data = f.create_group("data")
        total_episode_count = 0
        # Process each environment
        for environment, sim_or_real, count in environments:
            print(f"\nProcessing {environment} ({sim_or_real}) with {count} episodes")
            
            if download_from_s3:
                # Download episodes from S3 to local_base_path
                print ("here", os.path.join(local_base_path, task_name, environment, sim_or_real))
                if not os.path.exists(os.path.join(local_base_path, task_name, environment, sim_or_real)):
                    print (f"Downloading data for {task_name} in {environment} ({sim_or_real}) from S3...")
                    download_task_data_from_s3(
                        task_name, environment, sim_or_real, 
                        base_s3_path, local_base_path
                    )
            print ("Downloaded data, now processing...")
            
            type = 'bc/teleop/'
            local_task_path = os.path.join(local_base_path, task_name, environment, sim_or_real, type)
            if not os.path.exists(local_task_path):
                print(f"Local task path {local_task_path} does not exist, skipping environment")
                continue
            dates = os.listdir(local_task_path)
            date_paths = [os.path.join(local_task_path, date) for date in dates]
            
            print ("Local task path", local_task_path)
            episodes = []
            for date_path in date_paths:
                episode_names = os.listdir(date_path + "/diffusion_spartan/")
                processed = "processed"
                episodes.extend([os.path.join(date_path, "diffusion_spartan", episode_name, processed) for episode_name in episode_names if episode_name.startswith("episode_")])

            total_episode_count +=  len(episodes)
            # import ipdb; ipdb.set_trace()
            if len(episodes) == 0:
                print(f"No episodes found in {local_task_path}, skipping environment")
                continue
            
            
            # Prepare episode data for parallel processing
            episode_data_list = [(ep_idx, episode_path) for ep_idx, episode_path in enumerate(episodes) if os.path.exists(episode_path)]
            
            # Find optimal number of workers (run benchmark only once)
            # if max_workers is None and 'optimal_workers' not in locals():
            #     optimal_workers = benchmark_thread_count(episode_data_list)
            # elif max_workers is not None:
            #     optimal_workers = max_workers
            optimal_workers = 16
            
            print (f"Using {optimal_workers} worker threads for processing")
            # Process episodes in parallel
            with ThreadPoolExecutor(max_workers=optimal_workers) as executor:
                # Submit all episode processing tasks
                future_to_episode = {
                    executor.submit(process_episode_parallel, ep_data): ep_data 
                    for ep_data in episode_data_list
                }
                
                # Process completed tasks with progress bar
                processed_episodes = []
                for future in tqdm(future_to_episode, 
                                 desc=f"Processing {task_name} in {environment} ({sim_or_real})"):
                    result = future.result()
                    if result is not None:
                        processed_episodes.append(result)

            # DEBUG : Single thread processing
            # processed_episodes = []
            # for ep_data in tqdm(episode_data_list, desc=f"Processing {task_name} in {environment} ({sim_or_real})"):
            #     result = process_episode_parallel(ep_data)
            #     if result is not None:
            #         processed_episodes.append(result)
            
            # Store processed episodes in HDF5 (sequential to avoid conflicts)
            data.attrs["env_args"] = json.dumps({})
            for episode_result in processed_episodes:
                ep_idx = episode_result['ep_idx']
                group = data.create_group(f"demo_{ep_idx}")
                
                # Store data in HDF5
                group.create_dataset("actions_joints_act", 
                                   data=episode_result['combined_joint_act'])
                group.create_dataset("actions_xyz_act", 
                                   data=episode_result['combined_xyz_act'])
                group.attrs["num_samples"] = episode_result['num_samples']
                group.create_dataset("obs/front_img_1", 
                                   data=episode_result['front_img_1'])
                group.create_dataset("obs/ee_pose", 
                                   data=episode_result['ee_pose'])
                group.create_dataset("obs/joint_positions", 
                                   data=episode_result['joint_positions'])
                group.create_dataset("obs/intrinsics", 
                                   data=episode_result['intrinsics'])
                group.create_dataset("obs/extrinsics", 
                                   data=episode_result['extrinsics'])

        # Split train/validation
        f.close()

        if task_name not in OOD_TASKS:
            split_train_val_from_hdf5(hdf5_path=output_hdf5_path, val_ratio=VAL_RATIO)
        else:
            # this task only goes into val split
            split_train_val_from_hdf5(hdf5_path=output_hdf5_path, val_ratio=1.0)
        print(f"Completed processing {task_name} with {total_episode_count} episodes, saving at {output_hdf5_path}")
        
        # delete local data to save space
        if cleanup_local_data:
            local_task_dir = os.path.join(local_base_path, task_name)
            shutil.rmtree(local_task_dir)
            print(f"Deleted local data at {local_task_dir} to save space")

def benchmark_thread_count(episode_data_list, max_workers_to_test=None):
    """
    Benchmark different thread counts to find optimal performance
    """
    import time
    import multiprocessing
    
    if max_workers_to_test is None:
        max_workers_to_test = min(multiprocessing.cpu_count(), 16)
    
    # Test with small subset of episodes
    test_episodes = episode_data_list[:min(10, len(episode_data_list))]
    
    results = {}
    thread_counts = [1, 2, 4, 8, max_workers_to_test]
    
    for num_workers in thread_counts:
        print(f"Testing with {num_workers} workers...")
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(process_episode_parallel, ep_data) 
                      for ep_data in test_episodes]
            [future.result() for future in futures]  # Process all futures
        
        elapsed_time = time.time() - start_time
        results[num_workers] = elapsed_time
        print(f"  {num_workers} workers: {elapsed_time:.2f} seconds")
    
    # Find optimal thread count
    optimal_workers = min(results.keys(), key=results.get)
    print(f"\nOptimal thread count: {optimal_workers}")
    return optimal_workers

def main():
    parser = argparse.ArgumentParser(description='Process LBM simulation egocentric data to EgoMimic format')
    parser.add_argument('--csv_path', type=str, default='/home/swatigupta/Downloads/filtered_output.csv',
                        help='Path to the filtered_output.csv file')
    parser.add_argument('--base_s3_path', type=str, 
                        default='s3://robotics-manip-lbm/kylehatch/video_cotrain/HAMSTER_data/LBM_sim_egocentric/raw/data/tasks/{}/{}/{}/bc/teleop/{}/diffusion_spartan/episode_{}/processed/',
                        help='Base S3 path template for downloading data')
    parser.add_argument('--local_base_path', type=str, default='datasets/LBM_sim_egocentric/data/tasks',
                        help='Local base path for storing downloaded data')
    parser.add_argument('--output_base_path', type=str, default='datasets/LBM_sim_egocentric',
                        help='Base path for output HDF5 files')
    parser.add_argument('--download_from_s3', action='store_true', default=False,
                        help='Whether to download data from S3')
    parser.add_argument('--task_filter', type=str, nargs='*', default=None,
                        help='Specific tasks to process (if not provided, processes all tasks from CSV)')
    parser.add_argument('--cleanup_local_data', action='store_true', default=False,
                        help='Whether to delete local data after processing')
    parser.add_argument('--max_workers', type=int, default=None,
                        help='Maximum number of worker threads for parallel processing (auto-detect if not specified)')
    
    args = parser.parse_args()
    
    print(f"Processing tasks from CSV: {args.csv_path}")
    print(f"Base S3 path: {args.base_s3_path}")
    print(f"Local base path: {args.local_base_path}")
    print(f"Output base path: {args.output_base_path}")
    print(f"Download from S3: {args.download_from_s3}")
    print(f"Cleanup local data: {args.cleanup_local_data}")

    if args.task_filter:
        print(f"Task filter: {args.task_filter}")
        # Override the CSV loading to only process specified tasks
        original_load_tasks = load_tasks_from_csv
        def filtered_load_tasks(csv_path):
            all_tasks = original_load_tasks(csv_path)
            return [task for task in all_tasks if task in args.task_filter]
        
        # Temporarily replace the function
        import sys
        current_module = sys.modules[__name__]
        current_module.load_tasks_from_csv = filtered_load_tasks
    

    process_raw_data(
        csv_path=args.csv_path,
        base_s3_path=args.base_s3_path,
        local_base_path=args.local_base_path,
        output_base_path=args.output_base_path,
        download_from_s3=args.download_from_s3,
        cleanup_local_data=args.cleanup_local_data,
        max_workers=args.max_workers
    )

    # upload processed data  (egomimic format) to s3
    # s5cmd cp "datasets/LBM_sim_egocentric/converted/*" s3://robotics-manip-lbm/swatigupta/egomimic_data/LBM_sim_egocentric/processed/

if __name__ == "__main__":
    main()

# Train command: swatigupta@Puget-248656:~/EgoMimic/egomimic$
# python scripts/pl_train.py --config configs/egomimic_oboo.json --dataset ../datasets/LBM_sim_egocentric/converted/BimanualHangMugsOnMugHolderFromDryingRack.hdf5 --debug

# python egomimic/process_LBM_sim_egocentric_to_egomimic.py --task_filter BimanualPlacePearFromBowlOnCuttingBoard
# python egomimic/process_LBM_sim_egocentric_to_egomimic.py --task_filter BimanualPlacePearFromBowlOnCuttingBoard --download_from_s3  saved to -> datasets/LBM_sim_egocentric/converted/BimanualPlacePearFromBowlOnCuttingBoard.hdf5

# python egomimic/process_LBM_sim_egocentric_to_egomimic.py --download_from_s3 --cleanup_local_data  2>&1 | tee process_LBMsim_std_stderr2.txt

# python egomimic/process_LBM_sim_egocentric_to_egomimic.py --task_filter BimanualHangMugsOnMugHolderFromTable PutBananaOnSaucer PutOrangeInCenterOfTable --download_from_s3 --cleanup_local_data