from mimicplay.scripts.aloha_process.simarUtils import cam_frame_to_cam_pixels, draw_dot_on_frame, general_unnorm, miniviewer, nds, EXTRINSICS, WIDE_LENS_ROBOT_LEFT_K, aloha_fk, ee_pose_to_cam_frame
import torchvision
import numpy as np
import torch
import os
from mimicplay.algo.act import ACT
CURR_EXTRINSICS = EXTRINSICS["humanoidApr16"]

def visualize_ACT(preds, actions, frame):
    """
    preds: (T, 3) array of predicted actions
    actions: (T, 3) array of ground truth actions
    frame: (H, W, C) numpy array
    return frame with actions and preds drawn on it
    """


def evaluate_high_level_policy(model, data_loader, video_dir, max_samples=None):
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
        newvideo_dir = os.path.join(video_dir, f"eval_{vid_dir_count}")
        vid_dir_count += 1
    video_dir = newvideo_dir
    if not os.path.isdir(video_dir):
        os.mkdir(video_dir)
    
    metrics = {
        "paired_mse": [], # for each trajectory compute MSE((gt_t, gt_t+1), (pred_t, pred_t+1))
        "final_mse": [], # for each trajectory compute MSE(gt_t+T, pred_t+T)
    }
    #Internal realsense numbers
    intrinsics = WIDE_LENS_ROBOT_LEFT_K

    model.set_eval()

    count = 0
    vids_written = 0
    T = 700
    video = torch.zeros((T, 480, 640, 3))

    GOAL_COND = model.global_config.train.goal_mode

    for i, data in enumerate(data_loader):
        B = data["obs"]["front_img_1"].shape[0]
        if max_samples is not None and i * B > max_samples:
            break
        # import matplotlib.pyplot as plt
        # save_image(data["obs"]["front_img_1"][0, 0].numpy(), "/coc/flash9/skareer6/Projects/EgoPlay/EgoPlay/mimicplay/debug/image{i}.png")

        # save data["obs"]["front_img_1"][0, 0] which has type uint8 to file
        input_batch = model.process_batch_for_training(data)
        input_batch = model.postprocess_batch_for_training(input_batch, obs_normalization_stats=None) # TODO: look into obs norm
        if GOAL_COND and "ee_pose" in input_batch["goal_obs"]:
            del input_batch["goal_obs"]["ee_pose"]
        # del input_batch["actions"]
        info = model.forward_eval(input_batch)

        print(i)
        for b in range(B):
            im = data["obs"]["front_img_1"][b, 0].numpy()
            if isinstance(model, ACT):
                pred_values = info["actions"][b].cpu().numpy()
                actions = input_batch["actions"][b].cpu().numpy()
            else:
                pred_values = info.mean[b].view((10,3)).cpu().numpy()
                actions = input_batch["actions"][b, 0].view((10, 3)).cpu().numpy()

            if model.ac_key == "actions_joints":
                pred_values_drawable, actions_drawable = aloha_fk(pred_values[:, :6]), aloha_fk(actions[:, :6])
                pred_values_drawable, actions_drawable = ee_pose_to_cam_frame(pred_values_drawable, CURR_EXTRINSICS), ee_pose_to_cam_frame(actions_drawable, CURR_EXTRINSICS)
            else:
                pred_values_drawable, actions_drawable = pred_values, actions
            
            pred_values_drawable = cam_frame_to_cam_pixels(pred_values_drawable, intrinsics)
            actions_drawable = cam_frame_to_cam_pixels(actions_drawable, intrinsics)
            frame = draw_dot_on_frame(im, pred_values_drawable, show=False, palette="Purples")
            frame = draw_dot_on_frame(frame, actions_drawable, show=False, palette="Greens")

            add_metrics(metrics, actions, pred_values)

            if GOAL_COND:
                goal_frame = data["goal_obs"]["front_img_1"][b, 0].numpy()
                frame = miniviewer(frame, goal_frame)

            # cv2.imwrite(f"/coc/flash9/skareer6/Projects/EgoPlay/EgoPlay/mimicplay/debug/image{count}.png", frame)
            if count == T:
                if video_dir is not None:
                    torchvision.io.write_video(os.path.join(video_dir, f"_{vids_written}.mp4"), video[1:count], fps=30)
                # exit()
                count = 0
                vids_written += 1
                video = torch.zeros((T, 480, 640, 3))
            video[count] = torch.from_numpy(frame)

            count += 1
    
    torchvision.io.write_video(os.path.join(video_dir, f"_{vids_written}.mp4"), video[1:count], fps=30)
    # summarize metrics
    summary_metrics = {}
    for key in metrics:
        concat = np.stack(metrics[key], axis=0)
        mean_stat = np.mean(concat, axis=0)
        print(f"{key}: {mean_stat}")

        summary_metrics[key] = mean_stat

    if model.ac_key == "actions_joints":
        to_return = {
            "paired_mse_avg": np.mean(summary_metrics["paired_mse"]),
            "final_mse_avg": np.mean(summary_metrics["final_mse"]),
        }
    else:
        to_return = {
            "paired_mse x": summary_metrics["paired_mse"][0],
            "paired_mse y": summary_metrics["paired_mse"][1],
            "paired_mse z": summary_metrics["paired_mse"][2],
            "paired_mse_avg": np.mean(summary_metrics["paired_mse"]),
            "final_mse x": summary_metrics["final_mse"][0],
            "final_mse y": summary_metrics["final_mse"][1],
            "final_mse z": summary_metrics["final_mse"][2],
            "final_mse_avg": np.mean(summary_metrics["final_mse"]),
        }
        
    return to_return

def add_metrics(metrics, actions, pred_values):
    """
    metrics: {"paired_mse": [], "final_mse": []}
    actions: (10, 3) array of ground truth actions
    pred_values: (10, 3) array of predicted values
    """
    paired_mse = np.mean(np.square((pred_values - actions)*100), axis=0)
    final_mse = np.square((pred_values[-1] - actions[-1])*100)
    metrics["paired_mse"].append(paired_mse)
    metrics["final_mse"].append(final_mse)

    return metrics