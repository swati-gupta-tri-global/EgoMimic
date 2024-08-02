import h5py
import numpy as np
import argparse
import os
from tqdm import tqdm
import cv2
from mimicplay.scripts.aloha_process.simarUtils import nds, ee_pose_to_cam_frame, EXTRINSICS, ARIA_INTRINSICS, cam_frame_to_cam_pixels
# Example call: in aloha_process folder: python aloha_to_robomimic.py --dataset /coc/flash7/skareer6/calibrate_samples/ --arm left --out /coc/flash7/skareer6/calibrate_samples/ --task-name robomimic
import pytorch_kinematics as pk
import torch
from robomimic.scripts.split_train_val import split_train_val_from_hdf5
import json
import shutil
import matplotlib.pyplot as plt

"""
aloha_hdf5 has the following format
dict with keys:  <KeysViewHDF5 ['action', 'observations']>
action: (500, 14)
observations: dict with keys:  <KeysViewHDF5 ['effort', 'images', 'qpos', 'qvel']>
        effort: (500, 14)
        images: dict with keys:  <KeysViewHDF5 ['cam_high', 'cam_right_wrist']>
                cam_high: (500, 480, 640, 3)
                cam_right_wrist: (500, 480, 640, 3)
        qpos: (500, 14)
        qvel: (500, 14)
"""

def get_future_points(arr, POINT_GAP=15, FUTURE_POINTS_COUNT=10):
    future_traj = []

    for i in range(POINT_GAP, (FUTURE_POINTS_COUNT + 1) * POINT_GAP, POINT_GAP):
        # Identify the indices for the current and prior points
        index_current = min(len(arr) - 1, i)

        current_point = arr[index_current]
        future_traj.extend(current_point)

    return future_traj

def is_valid_path(path):
    return not os.path.isdir(path) and "episode" in path


def main(args):
    chain = pk.build_serial_chain_from_urdf(open("/coc/flash9/skareer6/Projects/EgoPlay/EgoPlay/mimicplay/scripts/aloha_process/model.urdf").read(), "vx300s/ee_gripper_link")

    with h5py.File(args.dataset, 'r+') as aloha_hdf5:
        for j in tqdm(range(0,len(aloha_hdf5['data']))):
            joint_pos = chain.forward_kinematics(torch.from_numpy(aloha_hdf5[f'data/demo_{j}/obs/joint_positions'][:,:-1]), end_only=False)
            
            fk_positions = joint_pos['vx300s/ee_gripper_link'].get_matrix()[:, :3, 3]
            elbow_positions = joint_pos['vx300s/upper_forearm_link'].get_matrix()[:, :3, 3]

            fk_positions = ee_pose_to_cam_frame(fk_positions, EXTRINSICS[args.extrinsics])[:, :3]
            elbow_positions = ee_pose_to_cam_frame(elbow_positions, EXTRINSICS[args.extrinsics])[:, :3]
            
            px_val_gripper = cam_frame_to_cam_pixels(fk_positions, ARIA_INTRINSICS)
            px_val_elbow = cam_frame_to_cam_pixels(elbow_positions, ARIA_INTRINSICS)

            line_images = np.zeros_like(aloha_hdf5[f'data/demo_{j}/obs/front_img_1'])
            mask_line_images = np.zeros_like(aloha_hdf5[f'data/demo_{j}/obs/front_img_1'])

            for i,image in enumerate(aloha_hdf5[f'data/demo_{j}/obs/front_img_1'][:]):
                image = cv2.line(image,(int(px_val_gripper[i,0]),int(px_val_gripper[i,1])),(int(px_val_elbow[i,0]),int(px_val_elbow[i,1])),color=(255,0,0), thickness=25)
                line_images[i] = image
                # plt.imsave("robo_overlay/masked.png", image)
                # breakpoint()

            aloha_hdf5[f'data/demo_{j}/obs'].create_dataset('front_img_1_line', data=line_images)
    
    print("Successful Conversion!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        help="path to source folder",
    )
    parser.add_argument(
        "--arm",
        type=str,
        help="which arm to convert data for"
    )
    parser.add_argument(
        "--extrinsics",
        type=str,
        help="which arm to convert data for"
    )
    args = parser.parse_args()

    main(args)

