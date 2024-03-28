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
    # breakpoint()

    vid_dir_count = 0
    newvideo_dir = video_dir
    # breakpoint()
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
    robot_count = 0
    vids_written = 0
    robot_vids_written = 0
    T = 400
    video = torch.zeros((T, 480, 640, 3))
    robot_video = torch.zeros((T, 480, 640, 3))

    for i, data in enumerate(data_loader):
        input_batch = model.process_batch_for_training(data)
        input_batch = model.postprocess_batch_for_training(input_batch, obs_normalization_stats=None) # TODO: look into obs norm

        has_robot_data = False
        required_keys = ['obs', 'goal_obs', 'actions', 'robot_obs', 'robot_goal_obs', 'robot_actions']

        if set(required_keys).issubset(input_batch.keys()):
            has_robot_data = True    
        
        # breakpoint()
        dummy = input_batch["obs"]["front_img_1"].permute(0, 2, 3, 1)
        dummy_frames = (255 * dummy.cpu().numpy()).astype(np.uint8)
        torchvision.io.write_video(os.path.join(video_dir, "dummy.mp4"), dummy_frames, fps=30)

        if has_robot_data:
            dummy_1 = input_batch["robot_obs"]["front_img_1"].permute(0, 2, 3, 1)
            dummy_frames_1 = (255 * dummy_1.cpu().numpy()).astype(np.uint8)
            torchvision.io.write_video(os.path.join(video_dir, "dummy_1.mp4"), dummy_frames_1, fps=30)


        if "ee_pose" in input_batch.get("goal_obs", {}):
            del input_batch["goal_obs"]["ee_pose"]
        if has_robot_data and "ee_pose" in input_batch.get("robot_goal_obs", {}):
            del input_batch["robot_goal_obs"]["ee_pose"]

        if "actions" in input_batch:
            del input_batch["actions"]
        if "robot_actions" in input_batch:
            del input_batch["robot_actions"]
        
        if not has_robot_data:
            info = model.forward_eval(input_batch)
        else:
            info, robot_info = model.forward_eval(input_batch)
        
        print(i)
        for b in range(data["obs"]["front_img_1"].shape[0]):
            im = data["obs"]["front_img_1"][b, 0].numpy()
            goal_frame = data["goal_obs"]["front_img_1"][b, 0].numpy()
            
            pred_values = np.ones((10, 3))
            for t in range(10):
                means = info.mean[b, t*3:3*(t+1)].cpu().numpy()
                px_val = cam_frame_to_cam_pixels(means[None, :], intrinsics)
                pred_values[t] = px_val

            frame = draw_dot_on_frame(im, pred_values, show=False, palette="Purples")

            actions = data["actions"][b, 0].view((10, 3))
            actions = actions.cpu().numpy()
            add_metrics(metrics, actions, info.mean[b].view((10,3)).cpu().numpy())
            for t in range(10):
                actions[t] = cam_frame_to_cam_pixels(actions[t][None, :], intrinsics)

            frame = draw_dot_on_frame(frame, actions, show=False, palette="Greens")

            frame = miniviewer(frame, goal_frame)

            if count == T:
                if video_dir is not None:
                    torchvision.io.write_video(os.path.join(video_dir, f"_{vids_written}.mp4"), video[1:count], fps=30)
                # exit()
                count = 0
                vids_written += 1
                video = torch.zeros((T, 480, 640, 3))
            video[count] = torch.from_numpy(frame)

            count += 1

        if has_robot_data:
            for b in range(data["robot_obs"]["front_img_1"].shape[0]):
                im = data["robot_obs"]["front_img_1"][b, 0].numpy()
                goal_frame = data["robot_goal_obs"]["front_img_1"][b, 0].numpy()
                
                pred_values = np.ones((10, 3))
                for t in range(10):
                    means = robot_info.mean[b, t*3:3*(t+1)].cpu().numpy()
                    px_val = cam_frame_to_cam_pixels(means[None, :], intrinsics)
                    pred_values[t] = px_val

                frame = draw_dot_on_frame(im, pred_values, show=False, palette="Purples")

                actions = data["robot_actions"][b, 0].view((10, 3))
                actions = actions.cpu().numpy()
                add_metrics(metrics, actions, robot_info.mean[b].view((10,3)).cpu().numpy())
                for t in range(10):
                    actions[t] = cam_frame_to_cam_pixels(actions[t][None, :], intrinsics)

                frame = draw_dot_on_frame(frame, actions, show=False, palette="Greens")

                frame = miniviewer(frame, goal_frame)

                if robot_count == T:
                    if video_dir is not None:
                        torchvision.io.write_video(os.path.join(video_dir, f"_{robot_vids_written}_robot.mp4"), robot_video[1:robot_count], fps=30)
                    # exit()
                    robot_count = 0
                    robot_vids_written += 1
                    robot_video = torch.zeros((T, 480, 640, 3))
                robot_video[robot_count] = torch.from_numpy(frame)

                robot_count += 1
    
    torchvision.io.write_video(os.path.join(video_dir, f"_{vids_written}.mp4"), video[1:count], fps=30)
    torchvision.io.write_video(os.path.join(video_dir, f"_{robot_vids_written}_robot.mp4"), robot_video[1:robot_count], fps=30)
        
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