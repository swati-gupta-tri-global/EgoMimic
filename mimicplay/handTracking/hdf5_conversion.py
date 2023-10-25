import h5py
import numpy as np
import cv2
import os 

hand_tracking_file = open("hand_tracking_data.txt", "r+")
hand_tracking_file.seek(0)


numData = len(hand_tracking_file.readlines())

poses = np.zeros((numData,7))
actions = np.zeros((numData,7))
count = 0
images = np.zeros((numData,480,640,3))

failed_to_read = False
prev_pose = None

hand_tracking_file.seek(0)

for idx, line in enumerate(hand_tracking_file.readlines()):
    pose = line.split()
    if os.path.isfile("images/frame_"+str(pose[0])):
        image = cv2.imread("images/frame_"+str(pose[0]))
        poses[idx]=np.array(pose[3:10],dtype=float)
        images[idx]=np.array(image)
        if(idx>=1):
            actions[idx]=np.array(poses[-1]-poses[-2])
    print("reading data...")

print("NumImages:", images.shape)
print("NumPoses:", poses.shape)
print("NumPrevPoses:", actions.shape)

# Create an HDF5 file
filename = "demo_data.h5"
with h5py.File(filename, "w") as f:
    # Create a group for demo_0
    demo_0 = f.create_group("demo_0")

    # Create datasets for the subgroups and data within demo_0
    obs = demo_0.create_group("obs")

    # Front_image_1: The hand tracking rgb camera + the low aloha camera
    front_image_1_data = images  # Example data, replace with actual data
    obs.create_dataset("Front_image_1", data=front_image_1_data)

    # Front_image_2: The high ALOHA camera
    front_image_2_data = np.random.rand(1, 1920, 1080, 3)
    obs.create_dataset("Front_image_2", data=front_image_2_data)

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

    actions = demo_0.create_group("actions")
    delta_ee_pose = actions
    actions.create_dataset(
        "delta_ee_pose", data=delta_ee_pose, dtype='float64')
