import numpy as np
import cv2
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import torch
from scipy.spatial.transform import Rotation
import pytorch_kinematics as pk
import egomimic
import os
import torchvision.transforms.v2.functional as TVTF
import scipy

REALSENSE_INTRINSICS = np.array(
    [[616.0, 0.0, 313.4, 0.0], [0.0, 615.7, 236.7, 0.0], [0.0, 0.0, 1.0, 0.0]]
)  # Internal realsense numbers
# K: [616.16650390625, 0.0, 313.42645263671875, 0.0, 615.7142333984375, 236.67532348632812, 0.0, 0.0, 1.0]

ARIA_INTRINSICS = np.array(
    [
        [133.25430222 * 2, 0.0, 320, 0],
        [0.0, 133.25430222 * 2, 240, 0],
        [0.0, 0.0, 1.0, 0],
    ]
)

# A2 paper without turning in skew direction
WIDE_LENS_ROBOT_LEFT_K = np.array(
    [
        [133.25430222 * 2, 0.0, 160.27941013 * 2, 0],
        [0.0, 133.2502574 * 2, 122.05743188 * 2, 0],
        [0.0, 0.0, 1.0, 0],
    ]
)
WIDE_LENS_HAND_LEFT_K = np.array(
    [
        [265.83575589493415, 0.0, 324.5832835740557, 0.0],
        [0.0, 265.8940770981264, 244.23118856728662, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ]
)

WIDE_LENS_ROBOT_LEFT_D = np.array(
    [[0.00087175, -0.00866803, 0.00016203, 0.00050252, -0.004487]]
)

# Cam to base extrinsics
EXTRINSICS = {
    "ariaJul29": {
        "left": np.array([[-0.02701913, -0.77838164,  0.62720969,  0.1222102 ],
       [ 0.99958387, -0.01469678,  0.02482135,  0.17666979],
       [-0.01010252,  0.62761934,  0.77845482,  0.00423704],
       [ 0.        ,  0.        ,  0.        ,  1.        ]]),

       "right": np.array([[ 0.07280155, -0.81760187,  0.57116295,  0.12038065],
       [ 0.9973441 ,  0.05843903, -0.04346979, -0.31690207],
       [ 0.00216277,  0.57281067,  0.81968485, -0.03742754],
       [ 0.        ,  0.        ,  0.        ,  1.        ]])
    },
    "ariaJul29L": np.array([[-0.02701913, -0.77838164,  0.62720969,  0.1222102 ],
       [ 0.99958387, -0.01469678,  0.02482135,  0.17666979],
       [-0.01010252,  0.62761934,  0.77845482,  0.00423704],
       [ 0.        ,  0.        ,  0.        ,  1.        ]]),
    "ariaJul29R": np.array([[ 0.07280155, -0.81760187,  0.57116295,  0.12038065],
       [ 0.9973441 ,  0.05843903, -0.04346979, -0.31690207],
       [ 0.00216277,  0.57281067,  0.81968485, -0.03742754],
       [ 0.        ,  0.        ,  0.        ,  1.        ]])

}


def is_key(x):
    return hasattr(x, "keys") and callable(x.keys)


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
    elif nested_ds is None:
        print("None")
    else:
        # print('\t' * (tab_level), end='')
        print(nested_ds.shape)

    if is_key(nested_ds):
        for key, value in nested_ds.items():
            print("\t" * (tab_level), end="")
            print(f"{key}: ", end="")
            nds(value, tab_level + 1)
    elif isinstance(nested_ds, list):
        print("\t" * tab_level, end="")
        print("Index[0]", end="")
        nds(nested_ds[0], tab_level + 1)


def ee_pose_to_cam_frame(ee_pose_base, T_cam_base):
    """
    ee_pose_base: (N, 3)
    T_cam_base: (4, 4)

    returns ee_pose_cam: (N, 3)
    """
    N, _ = ee_pose_base.shape
    ee_pose_base = np.concatenate([ee_pose_base, np.ones((N, 1))], axis=1)

    if T_cam_base.ndim == 2:  # Single transform (4, 4)
        ee_pose_grip_cam = np.linalg.inv(T_cam_base) @ ee_pose_base.T
        return ee_pose_grip_cam.T[:, :3]
    else:
        T_cam_inv = np.linalg.inv(T_cam_base)  # (N, 4, 4)
        # Batch matrix multiplication: (N, 4, 4) @ (N, 4, 1)
        ee_pose_grip_cam = np.einsum('nij,nj->ni', T_cam_inv, ee_pose_base)
        return ee_pose_grip_cam[:, :3]


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
    ee_pose_cam: (N, 3)
    intrinsics: 3x4 matrix
    """
    N, _ = ee_pose_cam.shape
    ee_pose_cam = np.concatenate([ee_pose_cam, np.ones((N, 1))], axis=1)
    # print("3d pos in cam frame: ", ee_pose_cam)

    # print("intrinsics: ", intrinsics.shape, ee_pose_cam.shape)
    px_val = intrinsics @ ee_pose_cam.T
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
        try:
            frame = cv2.circle(
                frame,
                (int(pixel_val[0]), int(pixel_val[1])),
                dot_size,
                color_palette[i],
                -1,
            )
        except:
            print("Got bad pixel_val: ", pixel_val)
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


def miniviewer(frame, goal_frame, location="top_right"):
    """
    overlay goal_frame in a corner of frame
    frame: (H, W, C) numpy array
    goal_frame: (H, W, C) numpy array
    location: "top_right", "top_left", "bottom_left", "bottom_right"

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
    if location == "top_right":
        frame[:, : goal_frame.shape[1], -goal_frame.shape[2] :] = goal_frame
    elif location == "top_left":
        frame[:, : goal_frame.shape[1], : goal_frame.shape[2]] = goal_frame
    elif location == "bottom_left":
        frame[:, -goal_frame.shape[1] :, : goal_frame.shape[2]] = goal_frame
    elif location == "bottom_right":
        frame[:, -goal_frame.shape[1] :, -goal_frame.shape[2] :] = goal_frame
    # frame[:, :goal_frame.shape[1], -goal_frame.shape[2]:] = goal_frame
    return frame.permute((1, 2, 0)).numpy()


def transformation_matrix_to_pose(T):
    R = T[:3, :3]
    p = T[:3, 3]
    rotation_quaternion = Rotation.from_matrix(R).as_quat()
    pose_array = np.concatenate((p, rotation_quaternion))
    return pose_array


class AlohaFK:
    def __init__(self):
        urdf_path = os.path.join(
            os.path.dirname(egomimic.__file__), "resources/model.urdf"
        )
        self.chain = pk.build_serial_chain_from_urdf(
            open(urdf_path).read(), "vx300s/ee_gripper_link"
        )

    def fk(self, qpos):
        if isinstance(qpos, np.ndarray):
            qpos = torch.from_numpy(qpos)

        return self.chain.forward_kinematics(qpos, end_only=True).get_matrix()[:, :3, 3]


def robo_to_aria_imstyle(im):
    im = TVTF.adjust_hue(im, -0.05)
    im = TVTF.adjust_saturation(im, 1.2)
    im = apply_vignette(im, exponent=1)

    return im


def create_vignette_mask(height, width, exponent=2):
    """
    Create a vignette mask with the given height and width.
    The exponent controls the strength of the vignette effect.
    """
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, height), torch.linspace(-1, 1, width), indexing="ij"
    )
    radius = torch.sqrt(x**2 + y**2) / 2
    mask = 1 - torch.pow(radius, exponent)
    mask = torch.clamp(mask, 0, 1)
    return mask


def apply_vignette(image_tensor, exponent=2):
    """
    Apply a vignette effect to a batch of image tensors.
    """
    N, C, H, W = image_tensor.shape
    vignette_mask = create_vignette_mask(H, W, exponent)
    vignette_mask = vignette_mask.unsqueeze(0).unsqueeze(
        0
    )  # Add batch and channel dimensions
    vignette_mask = vignette_mask.expand(
        N, C, H, W
    )  # Expand to match the batch of images
    vignette_mask = vignette_mask.to(image_tensor.device)
    return image_tensor * vignette_mask

def add_extra_train_splits(data, split_percentages):
    """
    data: hdf5 file in robomimic format
    split_percentages: list of percentages for each split, e.g. [0.7, 0.1, 0.2]
    add key "mask/train_{split_name}" which subsamples "mask/train" by split_percentages
    """
    N = len(data["mask/train"][:])
    random_order = np.random.permutation(N)
    mask = data["mask/train"][:]
    splits = []
    for split in split_percentages:
        # data[f"mask/train_{split_percentages:.2f}"] = random_order[:int(N*split)]
        sorted_order = np.sort(random_order[:int(N*split)])
        print(sorted_order)
        splits.append(sorted_order)
        print(mask[sorted_order])
        data[f"mask/train_{int(split*100)}%"] = mask[sorted_order]
    
    for i in range(4):
        print(i)
        assert set(splits[i]).issubset(set(splits[i+1]))

def interpolate_arr(v, seq_length):
    """
    v: (B, T, D)
    seq_length: int
    """
    assert len(v.shape) == 3
    if v.shape[1] == seq_length:
        return v
    
    interpolated = []
    for i in range(v.shape[0]):
        index = v[i]

        interp = scipy.interpolate.interp1d(
            np.linspace(0, 1, index.shape[0]), index, axis=0
        )
        interpolated.append(interp(np.linspace(0, 1, seq_length)))

    # size (B, seq_length, D)
    return np.array(interpolated)

def interpolate_keys(obs, keys, seq_length):
    """
    obs: dict with values of shape (T, D)
    keys: list of keys to interpolate
    seq_length: int changes shape (T, D) to (seq_length, D)
    """
    for k in keys:
        v = obs[k]
        L = v.shape[0]
        if L == seq_length:
            continue

        if k == "pad_mask":
            # interpolate it by simply copying each index (seq_length / seq_length_to_load) times
            obs[k] = np.repeat(v, (seq_length // L), axis=0)
        elif k != "pad_mask":
            interp = scipy.interpolate.interp1d(
                np.linspace(0, 1, L), v, axis=0
            )
            try:
                obs[k] = interp(np.linspace(0, 1, seq_length))
            except:
                raise ValueError(f"Interpolation failed for key: {k} with shape{k.shape}")