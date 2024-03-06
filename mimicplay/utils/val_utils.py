from mimicplay.scripts.aloha_process.simarUtils import cam_frame_to_cam_pixels, draw_dot_on_frame, general_unnorm, miniviewer, nds, EXTRINSICS, WIDE_LENS_ROBOT_LEFT_K
import torchvision
import numpy as np
import torch
import os

def evaluate_high_level_policy(model, data_loader, video_dir):
    """
    Evaluate high level trajectory prediciton policy.
    model: model loaded from checkpoint
    data_loader: validation data loader
    goal_distance: number of steps forward to predict
    video_path: path to save rendered video
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
    T = 200
    video = torch.zeros((T, 480, 640, 3))

    for i, data in enumerate(data_loader):
        # import matplotlib.pyplot as plt
        # save_image(data["obs"]["front_img_1"][0, 0].numpy(), "/coc/flash9/skareer6/Projects/EgoPlay/EgoPlay/mimicplay/debug/image{i}.png")

        # save data["obs"]["front_img_1"][0, 0] which has type uint8 to file
        input_batch = model.process_batch_for_training(data)
        input_batch = model.postprocess_batch_for_training(input_batch, obs_normalization_stats=None) # TODO: look into obs norm
        if "ee_pose" in input_batch.get("goal_obs", {}):
            del input_batch["goal_obs"]["ee_pose"]
        if "actions" in input_batch:
            del input_batch["actions"]
        info = model.forward_eval(input_batch)

        print(i)
        for b in range(data["obs"]["front_img_1"].shape[0]):
            im = data["obs"]["front_img_1"][b, 0].numpy()
            goal_frame = data["goal_obs"]["front_img_1"][b, 0].numpy()
            
            pred_values = np.ones((10, 3))
            for t in range(10):
                means = info.mean[b, t*3:3*(t+1)].cpu().numpy()
                # means = general_unnorm(means, -110.509903, 624.081421, -1, 1)
                # means[0] = general_unnorm(means[0], mins[0], maxs[0], -1, 1)
                # means[1] = general_unnorm(means[1], mins[1], maxs[1], -1, 1)
                # means[2] = general_unnorm(means[2], mins[2], maxs[2], -1, 1)
                px_val = cam_frame_to_cam_pixels(means[None, :], intrinsics)
                pred_values[t] = px_val

            frame = draw_dot_on_frame(im, pred_values, show=False, palette="Purples")

            actions = data["actions"][b, 0].view((10, 3))
            # actions[:, 0] = general_unnorm(actions[:, 0], mins[0], maxs[0], -1, 1)
            # actions[:, 1] = general_unnorm(actions[:, 1], mins[1], maxs[1], -1, 1)
            # actions[:, 2] = general_unnorm(actions[:, 2], mins[2], maxs[2], -1, 1)
            actions = actions.cpu().numpy()
            add_metrics(metrics, actions, info.mean[b].view((10,3)).cpu().numpy())
            for t in range(10):
                actions[t] = cam_frame_to_cam_pixels(actions[t][None, :], intrinsics)

            frame = draw_dot_on_frame(frame, actions, show=False, palette="Greens")

            # breakpoint()
            frame = miniviewer(frame, goal_frame)

            #import cv2
            #cv2.imwrite(f"/nethome/pmathur39/flash/EgoPlay/mimicplay/image{count}.png", frame)
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

    to_return = {
        "paired_mse x": summary_metrics["paired_mse"][0],
        "paired_mse y": summary_metrics["paired_mse"][1],
        "paired_mse z": summary_metrics["paired_mse"][2],
        "paired_mse avg": np.mean(summary_metrics["paired_mse"]),
        "final_mse x": summary_metrics["final_mse"][0],
        "final_mse y": summary_metrics["final_mse"][1],
        "final_mse z": summary_metrics["final_mse"][2],
        "final_mse avg": np.mean(summary_metrics["final_mse"]),
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