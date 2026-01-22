from builtins import ValueError, print
from egomimic.utils.egomimicUtils import (
    cam_frame_to_cam_pixels,
    draw_dot_on_frame,
    ee_pose_to_cam_frame,
)
# Import Panda FK for correct arm kinematics
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from panda_conversions import panda_fk_corrected
import torchvision
import numpy as np
import torch
import os
from egomimic.algo.act import ACT
from egomimic.algo.egomimic import EgoMimic
from tqdm import tqdm
import av
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean


def write_video_safe(filename, video_array, fps=30):
    """
    Write video with PyAV compatibility handling for different versions.
    
    Args:
        filename: Output video file path
        video_array: Tensor of shape (T, H, W, 3) with values in [0, 255]
        fps: Frames per second
    """
    try:
        # Try using torchvision's write_video directly
        torchvision.io.write_video(filename, video_array, fps=fps)
    except TypeError as e:
        if "an integer is required" in str(e):
            # PyAV 16+ requires manual video writing due to pict_type API change
            print(f"Warning: Using fallback video writer due to PyAV compatibility issue")
            
            # Convert to numpy and ensure uint8
            if isinstance(video_array, torch.Tensor):
                video_array = video_array.cpu().numpy()
            video_array = video_array.astype(np.uint8)
            
            # Open output container
            container = av.open(filename, mode='w')
            stream = container.add_stream('h264', rate=fps)
            stream.height = video_array.shape[1]
            stream.width = video_array.shape[2]
            stream.pix_fmt = 'yuv420p'
            
            # Write frames
            for frame_array in video_array:
                frame = av.VideoFrame.from_ndarray(frame_array, format='rgb24')
                for packet in stream.encode(frame):
                    container.mux(packet)
            
            # Flush stream
            for packet in stream.encode():
                container.mux(packet)
            
            container.close()
        else:
            raise

# EXTRINSICS_RIGHT = EXTRINSICS["ariaJul29"]["right"]
# EXTRINSICS_LEFT = EXTRINSICS["ariaJul29"]["left"]

# LBM extrinsics # Camera pose matrix T (4x4):
# [[-0.02281637 -0.76040487  0.64904841 -0.51159135]
#  [-0.99642311 -0.03554123 -0.07666683  0.10677921]
#  [ 0.08136581 -0.6484761  -0.75687407  0.7276687 ]
#  [ 0.          0.          0.          1.        ]]

# Rotation matrix R:
# [[-0.02281637 -0.76040487  0.64904841]
#  [-0.99642311 -0.03554123 -0.07666683]
#  [ 0.08136581 -0.6484761  -0.75687407]]

# Translation vector t:
# [-0.51159135  0.10677921  0.7276687 ]
LBM_EXTRINSICS_RIGHT = np.array([
    [-0.02281637, -0.76040487,  0.64904841, -0.51159135],
    [-0.99642311, -0.03554123, -0.07666683,  0.10677921],
    [ 0.08136581, -0.6484761 , -0.75687407,  0.7276687 ],
    [ 0.        ,  0.        ,  0.        ,  1.        ]
])
LBM_EXTRINSICS_LEFT = LBM_EXTRINSICS_RIGHT # fixed static camera

# LBM intrinsics
LBM_INTRINSICS = np.array([
    [[381., 0., 321.26901245, 0.],
     [0., 381., 247.86399841, 0.],
     [0., 0., 1., 0.]]
])
EGODEX_INTRINSICS = np.array([
    [[736.6339, 0., 960.],
     [0., 736.6339, 540.],
     [0., 0., 1.]]
])

# def draw_actions_on_frame(im, type, color, actions):
#     aloha_fk = AlohaFK()
#     if type == "joints": 
#         actions = aloha_fk.fk(actions[:, :6])
#         actions_drawable = ee_pose_to_cam_frame(actions, EXTRINSICS_RIGHT)
#     else:
#         actions_drawable = actions

#     actions_drawable = cam_frame_to_cam_pixels(actions_drawable, INTRINSICS)
#     im = draw_dot_on_frame(
#         im, actions_drawable, show=False, palette=color
#     )

#     return im

def draw_both_actions_on_frame(im, type, color, actions, arm="both", intrinsics=None, extrinsics=None, subsample=False):
    if intrinsics is None:
        raise ValueError("Intrinsics must be provided to draw actions on frame")
    
    if type == "joints": 
        # Optimize: Only print debug info occasionally to avoid performance hit
        # import random
        # debug_print = random.random() < 0.01  # Print 1% of the time
        # if debug_print:
        #     print(f"[DEBUG] Processing Panda joints actions with shape: {actions.shape}")
        
        # Performance optimization: For visualization, we don't need all 100 timesteps
        # Subsample to every 10th timestep to speed up FK computation
        if subsample and actions.shape[0] > 20:  # Only subsample if we have many timesteps
            step_size = max(1, actions.shape[0] // 10)  # Take ~10 samples
            sampled_indices = np.arange(0, actions.shape[0], step_size)
            actions_sampled = actions[sampled_indices]
        else:
            actions_sampled = actions
            sampled_indices = np.arange(actions.shape[0])
        
        if arm == "both":
            # Use Panda FK instead of ALOHA FK
            # For bimanual setup, assume 14-dim actions: [left_7, right_7]  
            if actions_sampled.shape[1] < 14:
                print(f"[WARNING] Expected 14 dimensions for bimanual, got {actions_sampled.shape[1]}")
                return im  # Return original image if dimensions don't match
                
            right_joint_positions = panda_fk_corrected(actions_sampled[:, 7:14])  # Exactly 7 joints
            right_actions = right_joint_positions['gripper']  # Already (N_sampled, 3)
            right_actions_drawable = ee_pose_to_cam_frame(right_actions, extrinsics)
            
            left_joint_positions = panda_fk_corrected(actions_sampled[:, :7])  # Exactly 7 joints
            left_actions = left_joint_positions['gripper']  # Already (N_sampled, 3)
            left_actions_drawable = ee_pose_to_cam_frame(left_actions, extrinsics)
            
            actions_drawable = np.concatenate((left_actions_drawable, right_actions_drawable), axis=0)
        elif arm == "right":
            # Use first 7 joints for right arm
            right_joint_positions = panda_fk_corrected(actions_sampled[:, :7])
            right_actions = right_joint_positions['gripper']  # Already (N_sampled, 3)
            right_actions_drawable = ee_pose_to_cam_frame(right_actions, extrinsics)
            actions_drawable = right_actions_drawable
        elif arm == "left":
            # Use first 7 joints for left arm  
            left_joint_positions = panda_fk_corrected(actions_sampled[:, :7])
            left_actions = left_joint_positions['gripper']  # Already (N_sampled, 3)
            left_actions_drawable = ee_pose_to_cam_frame(left_actions, extrinsics)
            actions_drawable = left_actions_drawable
    else:
        actions = actions.reshape(-1, 3)
        actions_drawable = actions
    
    actions_drawable = cam_frame_to_cam_pixels(actions_drawable, intrinsics)
    
    # Debug pixel coordinates occasionally
    import random
    if type != "joints" and random.random() < 0.01:  # Only for hand data, 1% of the time
        try:
            fx = intrinsics[0,0]
            fy = intrinsics[1,1]
            if hasattr(fx, 'item'):
                fx = fx.item()
            if hasattr(fy, 'item'):
                fy = fy.item()
            print(f"[DEBUG] Using intrinsics fx={fx:.1f}, fy={fy:.1f}")
        except:
            print(f"[DEBUG] Using intrinsics shape: {intrinsics.shape}")
        print(f"[DEBUG] Pixels {color} - range: [{np.min(actions_drawable):.1f}, {np.max(actions_drawable):.1f}]")
        valid_pixels = [(px[0], px[1]) for px in actions_drawable if 0 <= px[0] <= 640 and 0 <= px[1] <= 480]
        print(f"[DEBUG] Valid pixels in frame: {len(valid_pixels)}/{len(actions_drawable)}")
        if len(valid_pixels) > 0:
            print(f"[DEBUG] First few valid pixels: {valid_pixels[:3]}")
    
    im = draw_dot_on_frame(
        im, actions_drawable, show=False, palette=color
    )

    return im




def evaluate_high_level_policy(
    model, data_loader, video_dir, ac_key, max_samples=None, type=None,
):
    """
    Evaluate high level trajectory prediciton policy.
    model: model loaded from checkpoint
    data_loader: validation data loader
    goal_distance: number of steps forward to predict
    video_path: path to save rendered video
    acton_type: "xyz" or "joints"
    max_samples: maximum number of samples to evaluate
    """

    vid_dir_count = 0
    newvideo_dir = video_dir
    while os.path.isdir(newvideo_dir):
        newvideo_dir = os.path.join(video_dir, f"{type}_eval_{vid_dir_count}")
        vid_dir_count += 1
    video_dir = newvideo_dir
    os.makedirs(video_dir, exist_ok=True)

    metrics = {
        "paired_mse": [],  # for each trajectory compute MSE((gt_t, gt_t+1), (pred_t, pred_t+1))
        "final_mse": [],  # for each trajectory compute MSE(gt_t+T, pred_t+T)
        "path_distance": [],  # DTW distance between ground truth and predicted trajectories
    }

    count = 0
    vids_written = 0
    T = 700
    video = torch.zeros((T, 480, 640, 3))

    normalization_stats = data_loader.dataset.get_obs_normalization_stats()

    model.set_eval()

    front_cam_name = None
    for cam_name in model.global_config.observation.modalities.obs.rgb:
        if "front_img" in cam_name:
            front_cam_name = cam_name
    if front_cam_name is None:
        raise ValueError("Front camera not found in observation modalities.  Val utils expects that the main front camera key contains 'front_img'")

    # Check if video saving is enabled
    save_videos = model.global_config.experiment.get("save_eval_videos", True)  # Default to True for backward compatibility
    
    # Calculate total batches for progress bar
    total_batches = len(data_loader)
    if max_samples is not None:
        total_batches = min(total_batches, (max_samples + data_loader.batch_size - 1) // data_loader.batch_size)
    
    # Helper function to save intermediate stats
    def save_intermediate_stats(metrics, video_dir, type, ac_key, samples_processed):
        if len(metrics["paired_mse"]) == 0:
            return
        
        summary_metrics = {}
        for key in metrics:
            concat = np.stack(metrics[key], axis=0)
            mean_stat = np.mean(concat, axis=0)
            summary_metrics[key] = mean_stat
        
        if "joints" in ac_key:
            stats = {
                f"{type}_paired_mse_avg": np.mean(summary_metrics["paired_mse"]),
                f"{type}_final_mse_avg": np.mean(summary_metrics["final_mse"]),
                f"{type}_path_distance_avg": np.mean(summary_metrics["path_distance"]),
            }
        else:
            stats = {
                f"{type}_paired_mse x": summary_metrics["paired_mse"][0],
                f"{type}_paired_mse y": summary_metrics["paired_mse"][1],
                f"{type}_paired_mse z": summary_metrics["paired_mse"][2],
                f"{type}_paired_mse_avg": np.mean(summary_metrics["paired_mse"]),
                f"{type}_final_mse x": summary_metrics["final_mse"][0],
                f"{type}_final_mse y": summary_metrics["final_mse"][1],
                f"{type}_final_mse z": summary_metrics["final_mse"][2],
                f"{type}_final_mse_avg": np.mean(summary_metrics["final_mse"]),
                f"{type}_path_distance_avg": np.mean(summary_metrics["path_distance"]),
            }
        
        # Save to intermediate stats file
        stats_file = os.path.join(os.path.dirname(video_dir), f"step_log_intermediate_{samples_processed}_samples.txt")
        with open(stats_file, "w") as f:
            f.write(f"Samples processed: {samples_processed}\n")
            for k, v in stats.items():
                f.write(f"{k}: {v}\n")
        print(f"Saved intermediate stats to {stats_file}")
    
    samples_processed = 0
    save_interval = max(1, total_batches // 10)  # Save every 10% of progress
    
    for i, data in enumerate(tqdm(data_loader, total=total_batches, desc="Evaluating", unit="batch")):
        if isinstance(data, list):
            data = data[0]
        B = data[ac_key].shape[0]
        if max_samples is not None and i * B > max_samples:
            break
        # import matplotlib.pyplot as plt
        # save_image(data["obs"]["front_img_1"][0, 0].numpy(), "/coc/flash9/skareer6/Projects/EgoPlay/EgoPlay/mimicplay/debug/image{i}.png")

        # save data["obs"]["front_img_1"][0, 0] which has type uint8 to file
        input_batch = model.process_batch_for_training(data, ac_key)
        input_batch = model.postprocess_batch_for_training(
            input_batch, normalization_stats=normalization_stats, normalize_actions=False
        )  # TODO: look into obs norm
        unnorm_stats = normalization_stats if model.global_config.train.hdf5_normalize_actions else None

        info = model.forward_eval(input_batch, unnorm_stats=unnorm_stats)
        # print ("**************************Model pred keys: ", info.keys())
        # odict_keys(['actions_joints_act', 'actions_xyz_act'])
        
        for b in range(B):
            # Extract intrinsics from the dataset for correct projection
            dataset_intrinsics = None
            dataset_extrinsics = None
            if 'obs' in data and 'intrinsics' in data['obs']:
                dataset_intrinsics = data['obs']['intrinsics'][b, 0].cpu().numpy()  # [b, timestep] -> [3, 4]
                if dataset_intrinsics.shape == (3, 3):
                    # Expand to (3, 4)
                    dataset_intrinsics_expanded = np.zeros((3, 4))
                    dataset_intrinsics_expanded[:, :3] = dataset_intrinsics
                    dataset_intrinsics_expanded[:, 3] = [0, 0, 1]
                    dataset_intrinsics = dataset_intrinsics_expanded
                # Check if intrinsics are dummy (all zeros) - use hardcoded values instead
                if np.allclose(dataset_intrinsics[:3, :3], 0):
                    if type == "robot":
                        print(f"[DEBUG] Intrinsics are dummy (all zeros), using hardcoded LBM_INTRINSICS")
                        dataset_intrinsics = LBM_INTRINSICS
                    elif type == "hand":
                        print(f"[DEBUG] Intrinsics are dummy (all zeros), using hardcoded EGODEX_INTRINSICS")
                        dataset_intrinsics = EGODEX_INTRINSICS
                else:
                    print(f"[DEBUG] Using dataset intrinsics: fx={dataset_intrinsics[0,0]:.1f}, fy={dataset_intrinsics[1,1]:.1f}")
            elif type == "robot":
                # print(f"[DEBUG] No intrinsics found in batch for type {type}, using hardcoded LBM_INTRINSICS")
                dataset_intrinsics = LBM_INTRINSICS
            elif type == "hand":
                # print(f"[DEBUG] No intrinsics found in batch for type {type}, using hardcoded EGODEX_INTRINSICS")
                dataset_intrinsics = EGODEX_INTRINSICS
                
            if 'obs' in data and 'extrinsics' in data['obs']:
                dataset_extrinsics = data['obs']['extrinsics'][b, 0].cpu().numpy()  # [b, timestep] -> [4, 4]
                # Check if extrinsics are dummy (all zeros) - use hardcoded values instead
                if np.allclose(dataset_extrinsics, 0):
                    print(f"[DEBUG] Extrinsics are dummy (all zeros), using hardcoded LBM_EXTRINSICS_RIGHT")
                    dataset_extrinsics = LBM_EXTRINSICS_RIGHT
                else:
                    print(f"[DEBUG] Using dataset extrinsics from batch")
            else:
                # print(f"[DEBUG] No extrinsics found in batch for type {type}, using hardcoded LBM_EXTRINSICS_RIGHT/LEFT")
                dataset_extrinsics = LBM_EXTRINSICS_RIGHT
                
            
            im = (
                input_batch["obs"][front_cam_name][b].permute((1, 2, 0)).cpu().numpy()
                * 255
            ).astype(np.uint8)
            if isinstance(model, EgoMimic) or isinstance(model, ACT):
                if type == "robot":
                    actions = input_batch["actions_joints_act"][b].cpu().numpy()
                    pred_values = info["actions_joints_act"][b].cpu().numpy()

                    # actions = info["actions_xyz_act"][b].cpu().numpy()
                    # pred_values = input_batch["actions_xyz_act"][b].cpu().numpy()
                elif type == "hand":
                    actions = input_batch["actions_xyz_act"][b].cpu().numpy()
                    pred_values = info["actions_xyz_act"][b].cpu().numpy()
                else:
                    raise ValueError("type must be 'robot' or 'hand'")
            else:
                pred_values = info[ac_key][b].cpu().numpy()
                actions = input_batch[ac_key][b].cpu().numpy()

                # if pred_values.shape == (30,):
                #     pred_values = pred_values.reshape(-1, 3)
                # if actions.shape == (30,):
                #     actions = actions.reshape(-1, 3)

            ac_type = "joints" if "joints" in ac_key else "xyz"

            # im = draw_actions_on_frame(im, ac_type, "Greens", actions)
            # im = draw_actions_on_frame(im, ac_type, "Purples", pred_values)
            arm = "both"
            if actions.shape[1] == 14 or actions.shape[1] == 6:
                arm = "both"
            elif actions.shape[1] == 7 or actions.shape[1] == 3:
                arm = "right"

            # TODO(SWATI): debug, use correct extrinsics and intrinsics
            im = draw_both_actions_on_frame(im, ac_type, "Greens", actions, arm=arm, intrinsics=dataset_intrinsics, extrinsics=dataset_extrinsics)
            im = draw_both_actions_on_frame(im, ac_type, "Purples", pred_values, arm=arm, intrinsics=dataset_intrinsics, extrinsics=dataset_extrinsics)

            # (Swati)Disable drawing red dots for now (this is buggy)
            if isinstance(model, EgoMimic) and type == "robot":
                # im = draw_actions_on_frame(im, "xyz", "Reds", info["actions_xyz_act"][b].cpu().numpy())
                pred_actions_xyz = info["actions_xyz_act"][b].cpu().numpy()
                pred_actions_xyz = pred_actions_xyz.reshape(-1, 3)
                im = draw_both_actions_on_frame(im, "xyz", "Reds", pred_actions_xyz, arm=arm, intrinsics=dataset_intrinsics, extrinsics=dataset_extrinsics)

                # draw GT
                gt_actions_xyz = input_batch["actions_xyz_act"][b].cpu().numpy()
                gt_actions_xyz = gt_actions_xyz.reshape(-1, 3)
                im = draw_both_actions_on_frame(im, "xyz", "Greens", gt_actions_xyz, arm=arm, intrinsics=dataset_intrinsics, extrinsics=dataset_extrinsics)

            add_metrics(metrics, actions, pred_values)
            
            # Only save videos if enabled
            if save_videos:
                if count == T:
                    if video_dir is not None:
                        write_video_safe(
                            os.path.join(video_dir, f"{type}_{vids_written}.mp4"),
                            video[1:count],
                            fps=30,
                        )
                    # exit()
                    count = 0
                    vids_written += 1
                    video = torch.zeros((T, 480, 640, 3))
                video[count] = torch.from_numpy(im)

                count += 1
            samples_processed += 1
        
        # Save intermediate stats every 10% of progress
        if (i + 1) % save_interval == 0:
            save_intermediate_stats(metrics, video_dir, type, ac_key, samples_processed)

    # Save final video if enabled
    if save_videos and video_dir is not None:
        write_video_safe(
            os.path.join(video_dir, f"{type}_{vids_written}.mp4"), video[1:count], fps=30
        )
    # summarize metrics
    summary_metrics = {}
    for key in metrics:
        concat = np.stack(metrics[key], axis=0)
        mean_stat = np.mean(concat, axis=0)
        print(f"{key}: {mean_stat}")

        summary_metrics[key] = mean_stat

    if "joints" in ac_key:
        to_return = {
            f"{type}_paired_mse_avg": np.mean(summary_metrics["paired_mse"]),
            f"{type}_final_mse_avg": np.mean(summary_metrics["final_mse"]),
            f"{type}_path_distance_avg": np.mean(summary_metrics["path_distance"]),
        }
    else:
        to_return = {
            f"{type}_paired_mse x": summary_metrics["paired_mse"][0],
            f"{type}_paired_mse y": summary_metrics["paired_mse"][1],
            f"{type}_paired_mse z": summary_metrics["paired_mse"][2],
            f"{type}_paired_mse_avg": np.mean(summary_metrics["paired_mse"]),
            f"{type}_final_mse x": summary_metrics["final_mse"][0],
            f"{type}_final_mse y": summary_metrics["final_mse"][1],
            f"{type}_final_mse z": summary_metrics["final_mse"][2],
            f"{type}_final_mse_avg": np.mean(summary_metrics["final_mse"]),
            f"{type}_path_distance_avg": np.mean(summary_metrics["path_distance"]),
        }

    return to_return


def add_metrics(metrics, actions, pred_values):
    """
    metrics: {"paired_mse": [], "final_mse": [], "path_distance": []}
    actions: (seq_len, ac_dim) array of ground truth actions
    pred_values: (seq_len, ac_dim) array of predicted values
    """
    # print (f"[DEBUG] Actions shape: {actions.shape}, Pred values shape: {pred_values.shape}")
    paired_mse = np.mean(np.square((pred_values - actions) * 100), axis=0)
    final_mse = np.square((pred_values[-1] - actions[-1]) * 100)
    
    # Compute path distance using Dynamic Time Warping (DTW)
    path_distance, _ = fastdtw(actions, pred_values, dist=euclidean)
    
    metrics["paired_mse"].append(paired_mse)
    metrics["final_mse"].append(final_mse)
    metrics["path_distance"].append(path_distance)

    return metrics
