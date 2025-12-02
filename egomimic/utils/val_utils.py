from egomimic.utils.egomimicUtils import (
    cam_frame_to_cam_pixels,
    draw_dot_on_frame,
    general_unnorm,
    miniviewer,
    nds,
    EXTRINSICS,
    WIDE_LENS_ROBOT_LEFT_K,
    ARIA_INTRINSICS,
    ee_pose_to_cam_frame,
    AlohaFK,
    robo_to_aria_imstyle,
)
import torchvision
import numpy as np
import torch
import os
from egomimic.algo.act import ACT
from egomimic.algo.egomimic import EgoMimic
import scipy
from tqdm import tqdm
import av


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

EXTRINSICS_RIGHT = EXTRINSICS["ariaJul29"]["right"]
EXTRINSICS_LEFT = EXTRINSICS["ariaJul29"]["left"]

INTRINSICS = ARIA_INTRINSICS
EENORM = False
VIGNETTE = False
INTERP = False

def draw_actions_on_frame(im, type, color, actions):
    aloha_fk = AlohaFK()
    if type == "joints": 
        actions = aloha_fk.fk(actions[:, :6])
        actions_drawable = ee_pose_to_cam_frame(actions, EXTRINSICS_RIGHT)
    else:
        actions_drawable = actions

    actions_drawable = cam_frame_to_cam_pixels(actions_drawable, INTRINSICS)
    im = draw_dot_on_frame(
        im, actions_drawable, show=False, palette=color
    )

    return im

def draw_both_actions_on_frame(im, type, color, actions, arm="both"):
    aloha_fk = AlohaFK()
    if type == "joints": 
        if arm == "both":
            right_actions = aloha_fk.fk(actions[:, 7:13])
            right_actions_drawable = ee_pose_to_cam_frame(right_actions, EXTRINSICS_RIGHT)
            left_actions = aloha_fk.fk(actions[:, :6])
            left_actions_drawable = ee_pose_to_cam_frame(left_actions, EXTRINSICS_LEFT)
            actions_drawable = np.concatenate((left_actions_drawable, right_actions_drawable), axis=0)
        elif arm == "right":
            right_actions = aloha_fk.fk(actions[:, :6])
            right_actions_drawable = ee_pose_to_cam_frame(right_actions, EXTRINSICS_RIGHT)
            actions_drawable = right_actions_drawable
        elif arm == "left":
            left_actions = aloha_fk.fk(actions[:, :6])
            left_actions_drawable = ee_pose_to_cam_frame(left_actions, EXTRINSICS_LEFT)
            actions_drawable = left_actions_drawable
    else:
        actions = actions.reshape(-1, 3)
        actions_drawable = actions
    
    actions_drawable = cam_frame_to_cam_pixels(actions_drawable, INTRINSICS)
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
    }

    aloha_fk = AlohaFK()

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
        
        for b in range(B):
            im = (
                input_batch["obs"][front_cam_name][b].permute((1, 2, 0)).cpu().numpy()
                * 255
            ).astype(np.uint8)
            if isinstance(model, EgoMimic) or isinstance(model, ACT):
                if type == "robot":
                    actions = input_batch["actions_joints_act"][b].cpu().numpy()
                    pred_values = info["actions_joints_act"][b].cpu().numpy()
                elif type == "hand":
                    actions = input_batch["actions_xyz_act"][b].cpu().numpy()
                    pred_values = info["actions_xyz_act"][b].cpu().numpy()
                else:
                    raise ValueError("type must be 'robot' or 'hand'")
            else:
                pred_values = info[ac_key][b].cpu().numpy()
                actions = input_batch[ac_key][b].cpu().numpy()

                if pred_values.shape == (30,):
                    pred_values = pred_values.reshape(-1, 3)
                if actions.shape == (30,):
                    actions = actions.reshape(-1, 3)

            ac_type = "joints" if "joints" in ac_key else "xyz"

            # im = draw_actions_on_frame(im, ac_type, "Greens", actions)
            # im = draw_actions_on_frame(im, ac_type, "Purples", pred_values)
            arm = "both"
            if actions.shape[1] == 14 or actions.shape[1] == 6:
                arm = "both"
            elif actions.shape[1] == 7 or actions.shape[1] == 3:
                arm = "right"

            im = draw_both_actions_on_frame(im, ac_type, "Greens", actions, arm=arm)
            im = draw_both_actions_on_frame(im, ac_type, "Purples", pred_values, arm=arm)

            if isinstance(model, EgoMimic) and type == "robot":
                # im = draw_actions_on_frame(im, "xyz", "Reds", info["actions_xyz_act"][b].cpu().numpy())
                actions_xyz = info["actions_xyz_act"][b].cpu().numpy()
                actions_xyz = actions_xyz.reshape(-1, 3)
                im = draw_both_actions_on_frame(im, "xyz", "Reds", actions_xyz, arm=arm)
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
        }

    return to_return


def add_metrics(metrics, actions, pred_values):
    """
    metrics: {"paired_mse": [], "final_mse": []}
    actions: (10, 3) array of ground truth actions
    pred_values: (10, 3) array of predicted values
    """
    paired_mse = np.mean(np.square((pred_values - actions) * 100), axis=0)
    final_mse = np.square((pred_values[-1] - actions[-1]) * 100)
    metrics["paired_mse"].append(paired_mse)
    metrics["final_mse"].append(final_mse)

    return metrics
