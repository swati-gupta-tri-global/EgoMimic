import numpy as np
import cv2
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import torch
from scipy.spatial.transform import Rotation

REALSENSE_INTRINSICS = np.array([
    [616.0, 0.0, 313.4, 0.0],
    [0.0, 615.7, 236.7, 0.0],
    [0.0, 0.0, 1.0, 0.0]
]) #Internal realsense numbers
# K: [616.16650390625, 0.0, 313.42645263671875, 0.0, 615.7142333984375, 236.67532348632812, 0.0, 0.0, 1.0]


# Cam to base extrinsics
EXTRINSICS = {
    "table": np.array([
        [0.144, -0.598,  0.789, -0.017],
        [-0.978, 0.036, 0.206, -0.202],
        [-0.152, -0.801, -0.579, 0.491],
        [ 0.,     0.,     0.,     1.   ]
    ]),
    "humanoidJan19": np.array([
        [ 0.07, -0.911,  0.407, 0.035],
        [ 0.997,  0.048,   -0.063, -0.309],
        [ 0.038,  0.41, 0.911, -0.195],
        [0., 0., 0., 1.]
    ])
}

def is_key(x):
    return hasattr(x, 'keys') and callable(x.keys)

def is_listy(x):
    return isinstance(x, list)


def nds(nested_ds, tab_level=0):
    """
    Print the structure of a nested dataset.
    nested_ds: a series of nested dictionaries and iterables.  If a dictionary, print the key and recurse on the value.  If a list, print the length of the list and recurse on just the first index.  For other types, just print the shape.
    """
    # print('--' * tab_level, end='')
    if is_key(nested_ds):
        print("dict with keys: ", nested_ds.keys())
    elif is_listy(nested_ds):
        print("list of len: ", len(nested_ds))
    else:
        # print('\t' * (tab_level), end='')
        print(nested_ds.shape)

    if is_key(nested_ds):
        for key, value in nested_ds.items():
            print('\t' * (tab_level), end='')
            print(f"{key}: ", end="")
            nds(value, tab_level + 1)
    elif isinstance(nested_ds, list):
        print('\t' * tab_level, end='')
        print("Index[0]", end="")
        nds(nested_ds[0], tab_level+1)

def ee_pose_to_cam_frame(ee_pose_base, T_cam_base):
    """
    ee_pose_base: (N, 3)
    T_cam_base: (4, 4)

    returns ee_pose_cam: (N, 3)
    """
    N, _ = ee_pose_base.shape
    ee_pose_base = np.concatenate([ee_pose_base, np.ones((N, 1))], axis=1)

    ee_pose_grip_cam = np.linalg.inv(T_cam_base) @ ee_pose_base.T
    return ee_pose_grip_cam.T

def pose_transform(a_pose, T_a_b):
    """
    a_pose: (N, 3) series of poses in frame a
    T_a_b: (4, 4) transformation matrix from frame a to frame b

    returns b_pose: (N, 3) series of poses in frame b
    """
    orig_shape = list(a_pose.shape)
    a_pose = a_pose.reshape(-1, 3)
    N, _ = a_pose.shape
    a_pose = np.concatenate([a_pose, np.ones((N, 1))], axis=1)

    ee_pose_grip_cam = T_a_b @ a_pose.T
    orig_shape[-1] += 1
    return ee_pose_grip_cam.T.reshape(orig_shape)

def ee_pose_to_cam_pixels(ee_pose_base, T_cam_base, intrinsics):
    """
    ee_pose_base: (N, 3)
    T_cam_base: (4, 4)
    intrinsics: (3, 4)


    returns ee_pose_cam_pixels (N, 2)
    """
    N, _ = ee_pose_base.shape
    ee_pose_base = np.concatenate([ee_pose_base, np.ones((N, 1))], axis=1)

    ee_pose_grip_cam = np.linalg.inv(T_cam_base) @ ee_pose_base.T

    px_val = intrinsics @ ee_pose_grip_cam
    px_val = px_val / px_val[2, :]

    return px_val.T

def cam_frame_to_cam_pixels(ee_pose_cam, intrinsics):
    """
        camera frame 3d coordinates to pixels in camera frame
        ee_pose_cam: [x, y, z]
        intrinsics: 3x4 matrix
    """
    N, _ = ee_pose_cam.shape
    ee_pose_cam = np.concatenate([ee_pose_cam, np.ones((N, 1))], axis=1) 
    # print("intrinsics: ", intrinsics.shape, ee_pose_cam.shape)
    
    px_val = intrinsics @ ee_pose_cam.T
    # if not np.any(px_val):
    #     return px_val.T
    px_val = px_val / px_val[2, :]
    # print("2d pos cam frame: ", px_val)

    return px_val.T

def draw_dot_on_frame(frame, pixel_vals, show=True, palette="Purples", dot_size=5):
    """
    frame: (H, W, C) numpy array
    pixel_vals: (N, 2) numpy array of pixel values to draw on frame
    Drawn in light to dark order
    """
    frame = frame.astype(np.uint8).copy()
    if isinstance(pixel_vals, tuple):
        pixel_vals = [pixel_vals]

    # get purples color palette, and color the circles accordingly
    color_palette = plt.get_cmap(palette)
    color_palette = color_palette(np.linspace(0, 1, len(pixel_vals)))
    color_palette = (color_palette[:, :3] * 255).astype(np.uint8)
    color_palette = color_palette.tolist()


    for i, pixel_val in enumerate(pixel_vals):
        frame = cv2.circle(frame, (int(pixel_val[0]), int(pixel_val[1])), dot_size, color_palette[i], -1)
        if show:
            plt.imshow(frame)
            plt.show()

    return frame


def general_norm(array, min_val, max_val, arr_min=None, arr_max=None):
    if arr_min is None:
        arr_min = array.min()
    if arr_max is None:
        arr_max = array.max()
    
    return (max_val - min_val) * ((array - arr_min) / (arr_max - arr_min)) + min_val

def general_unnorm(array, orig_min, orig_max, min_val, max_val):
    return ((array - min_val) / (max_val - min_val)) * (orig_max - orig_min) + orig_min


def miniviewer(frame, goal_frame):
    """
    frame: (H, W, C) numpy array
    goal_frame: (H, W, C) numpy array
    
    return frame with goal_frame in top right corner (1/4 original size)

    resize using TF
    """
    frame = frame.copy()
    goal_frame = goal_frame.copy()
    if isinstance(frame, np.ndarray):
        frame = torch.from_numpy(frame)
    if isinstance(goal_frame, np.ndarray):
        goal_frame = torch.from_numpy(goal_frame)

    goal_frame = goal_frame.permute((2, 0, 1))
    frame = frame.permute((2, 0, 1))

    goal_frame = TF.resize(goal_frame, (frame.shape[1] // 4, frame.shape[2] // 4))
    frame[:, :goal_frame.shape[1], -goal_frame.shape[2]:] = goal_frame
    return frame.permute((1, 2, 0)).numpy()

def transformation_matrix_to_pose(T):
        R = T[:3, :3]
        p = T[:3, 3]
        rotation_quaternion = Rotation.from_matrix(R).as_quat()
        pose_array = np.concatenate((p, rotation_quaternion))
        return pose_array