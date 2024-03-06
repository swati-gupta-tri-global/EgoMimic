import os, cv2, shutil
import h5py
import json
import argparse
from robomimic.scripts.split_train_val import split_train_val_from_hdf5
from mimicplay.scripts.dataset_extract_traj_plans import process_dataset
import numpy as np
from mimicplay.scripts.aloha_process.simarUtils import general_norm, general_unnorm, ee_pose_to_cam_frame, nds, EXTRINSICS

# For either robot or hand data, must run this before training models via mimicplay.  Normalizes obs, chunks actions, and converts ee_pose to cam frame (robot)
# ex) python scripts/aloha_process/mimicplay_data_process.py --hdf5-path /coc/flash7/datasets/egoplay/stacking.hdf5 --data-type robot

parser = argparse.ArgumentParser()
parser.add_argument('--hdf5-path', type=str)
parser.add_argument('--data-type', type=str, choices=["hand", "robot"])
args = parser.parse_args()

def prep_for_mimicplay(hdf5_path, data_type):
    """
    dict with keys:  <KeysViewHDF5 ['data']>
    data: dict with keys:  <KeysViewHDF5 ['demo_0']>
        demo_0: dict with keys:  <KeysViewHDF5 ['actions', 'obs']>
            actions: (4047, ANYTHING)
            obs: dict with keys:  <KeysViewHDF5 ['ee_pose', 'front_image_1', 'front_image_2', 'gripper_position', 'wrist_cam_1']>
                ee_pose: (4047, 7)
                front_image_1: (4047, 480, 640, 3)
                front_image_2: (1, 1920, 1080, 3)
                gripper_position: ()
                wrist_cam_1: (1, 640, 480, 3)
        demo_1: ...
        demo_2: ...
    """
    COPY = True
    if COPY:
        target_path = hdf5_path.replace(".hdf5", "Mimicplay.hdf5")
        shutil.copy(hdf5_path, target_path)
        h5py_file = h5py.File(target_path, "r+")
    else:
        target_path = hdf5_path
        h5py_file = h5py.File(hdf5_path, "r+")

    add_data_dir(h5py_file)
    
    ##fix_demo_underscores(h5py_file)

    if data_type == "hand":
        print("Renaming front_image_1 and front_image_2 keys")

        key_dict = {'obs/front_image_1' : 'obs/front_img_1', 'obs/front_image_2' : 'obs/front_img_2'}

        replace_key_names(h5py_file, key_dict)

        scale_by_factor(h5py_file, 1)

        remove_corrupted_entries(h5py_file)

    if data_type == "hand":
        print("Renaming front_image_1 and front_image_2 keys")

        key_dict = {'obs/front_image_1' : 'obs/front_img_1', 'obs/front_image_2' : 'obs/front_img_2'}

        replace_key_names(h5py_file, key_dict)

        scale_by_factor(h5py_file, 100)

        remove_corrupted_entries(h5py_file)

    # NOTE: temp stub put back
    # if data_type == "hand":
    remove_eepose_quat(h5py_file)

    if data_type == "robot":
        base_to_cam(h5py_file, EXTRINSICS["humanoidFeb29R"])


    # normalize_obs(h5py_file)
    h5py_file["data"].attrs["env_args"] = json.dumps({}) # if no normalize obs

    if data_type == "robot":
        chunk_actions(h5py_file, POINT_GAP=15, FUTURE_POINTS_COUNT=10)
    elif data_type == "hand":
        chunk_actions(h5py_file, POINT_GAP=4, FUTURE_POINTS_COUNT=10)

    demo_keys = [key for key in h5py_file['data'].keys() if 'demo' in key]
    DEMO_COUNT = len(demo_keys)

    # set num samples for each demo in data
    for demo_key in demo_keys:
        h5py_file[f"data/{demo_key}"].attrs["num_samples"] = h5py_file[f"data/{demo_key}/actions"].shape[0]

        # It seems like robomimic wants (..., C, H, W) instead of (..., H, W, C)
        # for im_number in range(1, 3):
        #     im1 = h5py_file[f"data/{k}/obs/front_image_{im_number}"][...].transpose((0, 3, 1, 2))
        #     del h5py_file[f"data/{k}/obs/front_image_{im_number}"]
        #     dset = h5py_file.create_dataset(f"data/{k}/obs/front_image_{im_number}", data=im1)


    # # NOTE: REMOVE LATER, this is just so there's more than 1 demo
    # if "demo_1" not in h5py_file["data"].keys():
    #     h5py_file["data"]["demo_1"] = h5py_file["data"]["demo_0"]
    
    h5py_file.close()

    # Only have 1 demo so I'm going to spoof a second demo and use val ratio of 0.5
    # TODO: fix this ratio
    split_train_val_from_hdf5(hdf5_path=target_path, val_ratio=0.2, filter_key=None)


def normalize_obs(h5py_file):
    mins = np.array([np.inf, np.inf, np.inf])
    maxs = np.array([-np.inf, -np.inf, -np.inf])

    # for each demo, update the mins and maxs of obs/ee_pose (3d)
    demo_keys = [key for key in h5py_file['data'].keys() if 'demo' in key]
    DEMO_COUNT = len(demo_keys)

    for demo_key in demo_keys:
        # Extract the robot0_eef_pos data
        eef_pos = h5py_file[f'data/{demo_key}/obs/ee_pose'][...]
        mins = np.minimum(mins, np.min(eef_pos, axis=0))
        maxs = np.maximum(maxs, np.max(eef_pos, axis=0))

    # h5py_file["data/obs_mins"] = mins
    # h5py_file["data/obs_maxs"] = maxs
    h5py_file["data"].attrs["env_args"] = json.dumps({
        "obs_mins": mins.tolist(),
        "obs_maxs": maxs.tolist(),
    })


    for demo_key in demo_keys:
        ee_pose_unnorm = h5py_file[f"data/{demo_key}/obs/ee_pose"][:]
        ee_pose_unnorm[:, 0] = general_norm(ee_pose_unnorm[:, 0], -1, 1, arr_min=mins[0], arr_max=maxs[0])
        ee_pose_unnorm[:, 1] = general_norm(ee_pose_unnorm[:, 1], -1, 1, arr_min=mins[1], arr_max=maxs[1])
        ee_pose_unnorm[:, 2] = general_norm(ee_pose_unnorm[:, 2], -1, 1, arr_min=mins[2], arr_max=maxs[2])

        h5py_file[f"data/{demo_key}/obs/ee_pose_norm"] = ee_pose_unnorm

        # make ee_pose the normalized version, and call the unnormalized version ee_pose_unnorm
        h5py_file[f"data/{demo_key}/obs/ee_pose_unnorm"] = h5py_file[f"data/{demo_key}/obs/ee_pose"]
        del h5py_file[f"data/{demo_key}/obs/ee_pose"]
        h5py_file[f"data/{demo_key}/obs/ee_pose"] = h5py_file[f"data/{demo_key}/obs/ee_pose_norm"]
        del h5py_file[f"data/{demo_key}/obs/ee_pose_norm"]


def get_future_points(arr, POINT_GAP=4, FUTURE_POINTS_COUNT=10):
    future_traj = []

    for i in range(POINT_GAP, (FUTURE_POINTS_COUNT + 1) * POINT_GAP, POINT_GAP):
        # Identify the indices for the current and prior points
        index_current = min(len(arr) - 1, i)

        current_point = arr[index_current]
        future_traj.extend(current_point)

    return future_traj

def chunk_actions(h5py_file, POINT_GAP=15, FUTURE_POINTS_COUNT=10):
    # Open the HDF5 file in read+ mode (allows reading and writing)
    demo_keys = [key for key in h5py_file['data'].keys() if 'demo' in key]
    DEMO_COUNT = len(demo_keys)

    for demo_key in demo_keys:
        # Extract the robot0_eef_pos data
        eef_pos = h5py_file[f'data/{demo_key}/obs/ee_pose'][...]

        # Calculate the future trajectory for each data point
        future_traj_data = np.array([
            get_future_points(eef_pos[j:], POINT_GAP, FUTURE_POINTS_COUNT) for j in range(len(eef_pos))
        ])

        if "actions" in h5py_file[f"data/{demo_key}"].keys():
            del h5py_file[f"data/{demo_key}/actions"]

        # Create the new dataset
        h5py_file.create_dataset(f'data/{demo_key}/actions', data=future_traj_data)

    print(f"Processed {DEMO_COUNT} demos!")

def remove_eepose_quat(h5py_file):
    demo_keys = [key for key in h5py_file['data'].keys() if 'demo' in key]
    DEMO_COUNT = len(demo_keys)
    #breakpoint()

    for demo_key in demo_keys:
        h5py_file[f"data/{demo_key}/obs/ee_pose_full_unnorm"] = h5py_file[f"data/{demo_key}/obs/ee_pose"]
        del h5py_file[f"data/{demo_key}/obs/ee_pose"]
        h5py_file[f"data/{demo_key}/obs/ee_pose"] = h5py_file[f"data/{demo_key}/obs/ee_pose_full_unnorm"][:, :3]


def base_to_cam(h5py_file, T_cam_base):
    """
        h5py_file
        dict with keys:  <KeysViewHDF5 ['data']>
            data: dict with keys:  <KeysViewHDF5 ['demo_0']>
                demo_0: dict with keys:  <KeysViewHDF5 ['actions', 'obs']>
                    actions: (4047, ANYTHING)
                    obs: dict with keys:  <KeysViewHDF5 ['ee_pose', 'front_image_1', 'front_image_2', 'gripper_position', 'wrist_cam_1']>
                        ee_pose: (4047, 7)
                        front_image_1: (4047, 480, 640, 3)
                        front_image_2: (1, 1920, 1080, 3)
                        gripper_position: ()
                        wrist_cam_1: (1, 640, 480, 3)
                demo_1: ...
                demo_2: ...
        
        For each demo, convert ee_pose to base frame by calling ee_pose_to_cam_frame in simarUtils.py
        Don't need to convert actions, bc actions are set later
    """
    demo_keys = [key for key in h5py_file['data'].keys() if 'demo' in key]
    DEMO_COUNT = len(demo_keys)


    # R_cam_base = np.array([
    #     [ 0.144, -0.598, 0.789],
    #     [-0.978, 0.036, 0.206],
    #     [-0.152, -0.801, -0.579]
    # ])
    # Tr_cam_base = np.array([[-0.017, -0.202, 0.491]])
    # T_cam_base = np.concatenate([R_cam_base, Tr_cam_base.T], axis=1)
    # T_cam_base = np.concatenate([T_cam_base, np.array([[0, 0, 0, 1]])], axis=0)
    # print("WARNING: using hardcoded T_cam_base")

    for demo_key in demo_keys:
        h5py_file[f"data/{demo_key}/obs/ee_pose_cam_frame"] = ee_pose_to_cam_frame(h5py_file[f"data/{demo_key}/obs/ee_pose"][:], T_cam_base)[:, :3]
        del h5py_file[f"data/{demo_key}/obs/ee_pose"]
        h5py_file[f"data/{demo_key}/obs/ee_pose"] = h5py_file[f"data/{demo_key}/obs/ee_pose_cam_frame"]
        del h5py_file[f"data/{demo_key}/obs/ee_pose_cam_frame"]

        # TODO: for low level policy training, convert the full xyzquat, not just xyz


def fix_demo_underscores(h5py_file):
    """
    h5py_file: 
        data
            demo0
            demo1
    
    for each demo above, change demo0 to demo_0 and demo1 to demo_1
    """
    demo_keys = [key for key in h5py_file['data'].keys() if 'demo' in key]

    for demo_key in demo_keys:
        if "_" not in demo_key:
            new_demo_key = demo_key.replace("demo", "demo_")
            h5py_file[f"data/{new_demo_key}"] = h5py_file[f"data/{demo_key}"]
            del h5py_file[f"data/{demo_key}"]

def replace_key_names(h5py_file, key_dict):
    """
        key_dict = {old_key : new_key }
    """
    demo_keys = [key for key in h5py_file['data'].keys() if 'demo' in key]

    for demo_key in demo_keys:
        print(f"replacing keys in {demo_key}")
        for old_key in key_dict:
            new_key = key_dict[old_key]
            if f"data/{demo_key}/{old_key}" in h5py_file:
                group = h5py_file[f"data/{demo_key}"]
                data = group[old_key][()]
                group.create_dataset(new_key, data=data)
                del group[old_key]

def scale_by_factor(h5py_file, factor):
    demo_keys = [key for key in h5py_file['data'].keys() if 'demo' in key]
    for demo_key in demo_keys:
        print(f"scaling by factor in {demo_key}")
        # actions
        actions = h5py_file[f"data/{demo_key}/actions"]
        actions[:] =  actions[:] / factor
        actions.flush()

        # obs/ee_pose
        ee_pose = h5py_file[f"data/{demo_key}/obs/ee_pose"]
        ee_pose[:] = ee_pose[:] / factor
        ee_pose.flush()

def add_data_dir(h5py_file):
    """
    h5py_file: 
        demo0
        demo1
    returns:
        data
            demo0
            demo1
    """
    if "data" not in h5py_file:
        h5py_file.create_group("data")
        for key in h5py_file.keys():
            if "demo" in key:
                h5py_file[f"data/{key}"] = h5py_file[key]
                del h5py_file[key]

def remove_corrupted_entries(h5py_file):
    demo_keys = [key for key in h5py_file['data'].keys() if 'demo' in key]
    for demo_key in demo_keys:
        print(f"Removing corrupted entries in {demo_key}")
        front_img_1 = h5py_file[f'data/{demo_key}/obs/front_img_1'][:]
        ee_pose = h5py_file[f'data/{demo_key}/obs/ee_pose'][:]
        actions = h5py_file[f'data/{demo_key}/actions'][:]
        
        zero_indices = [i for i, img in enumerate(front_img_1) if not np.any(img)]
        
        filtered_front_img_1 = np.delete(front_img_1, zero_indices, axis=0)
        filtered_ee_pose = np.delete(ee_pose, zero_indices, axis=0)
        filtered_actions = np.delete(actions, zero_indices, axis=0)

        del h5py_file[f'data/{demo_key}/obs/front_img_1']
        del h5py_file[f'data/{demo_key}/obs/ee_pose']
        del h5py_file[f'data/{demo_key}/actions']

        img_shape = front_img_1.shape

        chunks = (1,) + img_shape[1:]
        try:
            h5py_file.create_dataset(f'data/{demo_key}/obs/front_img_1', data=filtered_front_img_1, chunks=chunks)
            h5py_file.create_dataset(f'data/{demo_key}/obs/ee_pose', data=filtered_ee_pose)
            h5py_file.create_dataset(f'data/{demo_key}/actions', data=filtered_actions)
        except:
            print("Could not remove corrupted entries!")
<<<<<<< HEAD
        
=======

>>>>>>> main
if __name__ == '__main__':
    prep_for_mimicplay(args.hdf5_path, args.data_type)