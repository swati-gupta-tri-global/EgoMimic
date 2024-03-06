import h5py
import numpy as np
from modern_robotics import FKinSpace
import os
from mimicplay.scripts.aloha_process.simarUtils import transformation_matrix_to_pose

def single_episode_conversion(filepath, arm, output_dir):
    """
        filepath: full path to hdf5 to convert
        arm: arm to convert
        output_path: full path to converted location
    """
    output_path = os.path.join(output_dir, os.path.basename(filepath))
    output_path = output_path.split(".")[0] + "_out.hdf5"
    # if output_path == None:
    #     # just append out to file name
    #     output_path = filepath.split(".")[0] + "_out.hdf5"

    M = np.array([[1.0, 0.0, 0.0, 0.536494],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.42705],
                [0.0, 0.0, 0.0, 1.0]])

    Slist = np.array([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, -0.12705, 0.0, 0.0],
                    [0.0, 1.0, 0.0, -0.42705, 0.0, 0.05955],
                    [1.0, 0.0, 0.0, 0.0, 0.42705, 0.0],
                    [0.0, 1.0, 0.0, -0.42705, 0.0, 0.35955],
                    [1.0, 0.0, 0.0, 0.0, 0.42705, 0.0]]).T

    file = h5py.File(filepath, "r")

    obs = file['observations']
    actions = file['action']
    actions_qpos = np.array(actions)
    obs_qpos = np.array(obs['qpos'])
    obs_qvel = np.array(obs['qvel'])

    cam_front1 = np.array(obs['images/cam_high'])
    if 'images/cam_low' in obs:
        cam_front2 = np.array(obs['images/cam_low'])


    if arm == 'left' :
        arm_actions_qpos = actions_qpos[:, :6]
        cam_wrist = np.array(obs['images/cam_left_wrist'])
        arm_obs_qpos = obs_qpos[:, :6]
        arm_gripper_qpos = obs_qpos[:, 6]
        arm_gripper_qvel = obs_qvel[:, :6]

    elif arm == 'right' :
        arm_actions_qpos = actions_qpos[:, 7:13]
        cam_wrist = np.array(obs['images/cam_right_wrist'])
        arm_obs_qpos = obs_qpos[:, 7:13]
        arm_gripper_qpos = obs_qpos[:, 13]
        arm_gripper_qvel = obs_qvel[:, 7:13]
    else:
        print("Invalid arm argument!")
        exit()

    T_actions = np.zeros((arm_actions_qpos.shape[0], 4, 4))
    T_obs = np.zeros((arm_obs_qpos.shape[0], 4, 4))

    actions_ee_pose = np.zeros((arm_actions_qpos.shape[0], 7))
    obs_ee_pose = np.zeros((arm_obs_qpos.shape[0], 7))

    for i in range(arm_actions_qpos.shape[0]):
        # if i == 200:
        #     breakpoint()

        # T_actions[i] = FKinSpace(M, Slist, arm_actions_qpos[i])
        T_obs[i] = FKinSpace(M, Slist, arm_obs_qpos[i])

        if i != 0:
            # actions_ee_pose[i-1] = transformation_matrix_to_pose(T_actions[i])
            actions_ee_pose[i-1] = transformation_matrix_to_pose(
                T_obs[i] @ np.linalg.inv(T_obs[i-1])
            )

        obs_ee_pose[i] = transformation_matrix_to_pose(T_obs[i])

    if os.path.exists(output_path):
        print("Helper: output path already exists!")
        exit()
        # os.remove(filepath.split(".")[0] + "_out.hdf5")

    assert cam_front1.shape[1:] == (480, 640, 3)
    if 'images/cam_low' in obs:
        assert cam_front2.shape[1:] == (480, 640, 3)
    assert cam_wrist.shape[1:] == (480, 640, 3)
    with h5py.File(output_path, 'w', rdcc_nbytes=1024**2*2) as f:
        actions_group = f.create_group('actions')
        obs_group = f.create_group('obs')

        actions_group.create_dataset('ee_pose', data=actions_ee_pose)

        obs_group.create_dataset('ee_pose', data=obs_ee_pose)
        obs_group.create_dataset('front_img_1', data=cam_front1, dtype='uint8', chunks=(1, 480, 640, 3))
        if 'images/cam_low' in obs:
            obs_group.create_dataset('front_img_2', data=cam_front2, dtype='uint8', chunks=(1, 480, 640, 3))
        obs_group.create_dataset('wrist_img_1', data=cam_wrist, dtype='uint8', chunks=(1, 480, 640, 3))
        obs_group.create_dataset('gripper_position', data=arm_gripper_qpos)
        obs_group.create_dataset('joint_positions', data=arm_obs_qpos)
        obs_group.create_dataset('joint_vel', data=arm_gripper_qvel)

    print("Successful Conversion of " + filepath)


# def main():
#     single_episode_conversion("/coc/flash7/skareer6/touchPoints/touch_points.hdf5", "left")

# main()