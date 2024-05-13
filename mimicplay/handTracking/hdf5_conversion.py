import h5py
import numpy as np
import cv2
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--demo", type=str, help="demo number")

args = parser.parse_args()

demoNumber = args.demo

hand_tracking_file = open(f'build/hand_tracking_data{demoNumber}.txt', "r+")
hand_tracking_file.seek(0)


numData = len(hand_tracking_file.readlines())

poses = np.zeros((numData,7))
actions = np.zeros((numData,7))
count = 0
images = np.zeros((numData,480,640,3))

failed_to_read = False
prev_pose = None

hand_tracking_file.seek(0)

def array_into_chunks(arr, chunkSize):
    """
    Given an array of vectors (T, k) ex) [[1, 2], [3, 4], [5, 6], [7, 8]]
    and a chunkSize ex) 3
    return a list of chunks with padding (T, k*3) ex) [[1, 2, 3, 4, 5, 6], [3, 4, 5, 6, 7, 8], [5, 6, 7, 8, 7, 8], [7, 8, 7, 8, 7, 8]]
    """
    chunked_arr = []
    for i in range(arr.shape[0]):
        chunked_arr.append(np.concatenate([arr[i:], np.tile(arr[-1:], (chunkSize-1, 1))], axis=0)[:chunkSize].flatten())
    return np.array(chunked_arr)

count = 0
for idx, line in enumerate(hand_tracking_file.readlines()):
    pose = line.split()
    if os.path.isfile(f"build/demo{demoNumber}/frame_"+str(pose[0])):
        try:
            image = cv2.imread(f"build/demo{demoNumber}/frame_"+str(pose[0]))
            # Transform coordindates and populate pose
            poses[count][0]=-np.array(pose[3],dtype=float)
            poses[count][1]=np.array(pose[5],dtype=float)-175
            poses[count][2]=np.array(pose[4],dtype=float)
            poses[count][3:] = np.array(pose[6:10])
            images[count]=np.array(image)
            print(f"Index: {count}           Position: {poses[count][0:3]}")
        except:
            print("EOF")
        count+=1

actions = np.array(array_into_chunks(poses,5))

print("NumImages:", images.shape)
print("NumPoses:", poses.shape)
print("Actions:", actions.shape)

# Create an HDF5 file
filename = f"demo{demoNumber}.h5"
with h5py.File(filename, "w") as f:
    # Create a group for demo_0
    data = f.create_group("data")
    demo_0 = data.create_group("demo_0")

    # Create datasets for the subgroups and data within demo_0
    obs = demo_0.create_group("obs")

    # Front_image_1: The hand tracking rgb camera + the low aloha camera
    front_image_1_data = images  # Example data, replace with actual data
    obs.create_dataset("front_image_1", data=front_image_1_data)

    # Front_image_2: The high ALOHA camera
    front_image_2_data = np.random.rand(1, 1920, 1080, 3)
    obs.create_dataset("front_image_2", data=front_image_2_data)

    # wrist_cam_1: aloha wrist cam
    # Example data, replace with actual data
    wrist_cam_1_data = np.random.rand(1, 640, 480, 3)
    obs.create_dataset("wrist_cam_1", data=wrist_cam_1_data)

    # ee_pose: 7 xyz + quat. Use this for BOTH the hand tracking data, as well as the on robot demonstrations
    ee_pose_data = poses
    obs.create_dataset("ee_pose", data=ee_pose_data)

    # Gripper_position: 1d scalar
    gripper_position_data = 0.5  # Example data, replace with actual data
    obs.create_dataset("Gripper_position",
                       data=gripper_position_data, dtype='float64')

    # actions
    demo_0.create_dataset("actions", data=actions, dtype='float64')