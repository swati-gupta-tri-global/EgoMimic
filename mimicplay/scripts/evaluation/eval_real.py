"""
The main entry point for training policies.

Args:
    config (str): path to a config json that will be used to override the default settings.
        If omitted, default settings are used. This is the preferred way to run experiments.

    algo (str): name of the algorithm to run. Only needs to be provided if @config is not
        provided.

    name (str): if provided, override the experiment name defined in the config

    dataset (str): if provided, override the dataset path defined in the config

    bddl_file (str): if provided, the task's goal is specified as the symbolic goal in the bddl file (several symbolic predicates connected with AND / OR)

    video_prompt (str): if provided, a task video prompt is loaded and used in the evaluation rollouts

    debug (bool): set this flag to run a quick training run for debugging purposes
"""

import argparse
import json
import numpy as np
import scipy
import time
import os
import psutil
import sys
import traceback

from collections import OrderedDict

import torch
from torch.utils.data import DataLoader

import robomimic.utils.train_utils as TrainUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
from robomimic.utils.log_utils import PrintLogger, DataLogger
from torchvision.utils import save_image
import cv2
from interbotix_common_modules.common_robot.robot import (
    create_interbotix_global_node,
    robot_shutdown,
    robot_startup,
)
from mimicplay.scripts.evaluation.norm_stats import NORM_STATS
from aloha.constants import DT

PUPPET_GRIPPER_JOINT_OPEN = 1.0

from mimicplay.scripts.aloha_process.simarUtils import (
    cam_frame_to_cam_pixels,
    draw_dot_on_frame,
    general_unnorm,
    miniviewer,
    nds,
    ARIA_INTRINSICS,
    EXTRINSICS,
    ee_pose_to_cam_frame,
    AlohaFK,
)
import torchvision


from mimicplay.configs import config_factory
from mimicplay.pl_utils.pl_model import ModelWrapper
import datetime

from aloha.robot_utils import move_grippers  # requires aloha
from aloha.real_env import make_real_env  # requires aloha

from mimicplay.scripts.evaluation.real_utils import *
import matplotlib.pyplot as plt
from mimicplay.algo.act import ACT

from IPython.core import ultratb
import sys

# For debugging
# sys.excepthook = ultratb.FormattedTB(mode="Plain", color_scheme="Neutral", call_pdb=1)


CURR_INTRINSICS = ARIA_INTRINSICS
CURR_EXTRINSICS = EXTRINSICS["ariaJul29R"]
CAMERA_NAMES = ["cam_high", "cam_right_wrist"]
# NORM_STATS = to_torch(NORM_STATS, torch.device("cuda"))


def eval_real(model, config, env, rollout_dir):
    device = torch.device("cuda")

    aloha_fk = AlohaFK()

    # query_frequency = policy_config['num_queries']
    # if temporal_agg:
    #     query_frequency = 1
    #     num_queries = policy_config['num_queries']
    query_frequency = 100

    # max_timesteps = int(max_timesteps * 1)  # may increase for real-world tasks

    num_rollouts = 50
    for rollout_id in range(num_rollouts):
        rollout_id += 0

        ts = env.reset()

        t0 = time.time()
        with torch.inference_mode():
            rollout_images = []
            for t in range(700):
                obs = ts.observation
                # plt.imsave(os.path.join(rollout_dir, f"viz{t}.png"), obs["images"]["cam_high"])
                # plt.imsave(os.path.join(rollout_dir, f"wrist{t}.png"), obs["images"]["cam_right_wrist"])

                qpos = np.array(obs["qpos"])
                qpos = torch.from_numpy(qpos).float().unsqueeze(0).to(device)

                ### query policy
                if t % query_frequency == 0:
                    data = {
                        "obs": {
                            "front_img_1": (
                                torch.from_numpy(obs["images"]["cam_high"][None, None, :])
                            ).to(torch.uint8),
                            "right_wrist_img": (
                                torch.from_numpy(obs["images"]["cam_right_wrist"][None, None, :])
                            ).to(torch.uint8),
                            # "ee_pose": torch.from_numpy(ee_pose_cam_frame), #torch.tensor([[[0.2889, 0.1556, 0.4028]]]), #ee_pose_input[:, None, :], #TODO: Switch this to actual qpos (and make corresponding change in config)
                            "pad_mask": torch.ones((1, 100, 1)).to(device).bool(),
                            "joint_positions": qpos[..., 7:].reshape((1, 1, -1)),
                        },
                        "type": torch.tensor([0]),
                    }

                    # postprocess_batch
                    input_batch = model.process_batch_for_training(
                        data, "actions_joints"
                    )
                    input_batch["obs"]["front_img_1"] = input_batch["obs"][
                        "front_img_1"
                    ].permute(0, 3, 1, 2)
                    input_batch["obs"]["right_wrist_img"] = input_batch["obs"][
                        "right_wrist_img"
                    ].permute(0, 3, 1, 2)
                    input_batch["obs"]["front_img_1"] /= 255.0
                    input_batch["obs"]["right_wrist_img"] /= 255.0
                    input_batch = ObsUtils.normalize_batch(input_batch, normalization_stats=NORM_STATS, normalize_actions=False)
                    info = model.forward_eval(input_batch, unnorm_stats=NORM_STATS)


                    # Draw Actions
                    im = data["obs"]["front_img_1"][0, 0].cpu().numpy()
                    pred_values = info["actions"][0].cpu().numpy()

                    if "joints" in model.ac_key:
                        pred_values_drawable = aloha_fk.fk(pred_values[:, :6])
                        pred_values_drawable = ee_pose_to_cam_frame(pred_values_drawable, CURR_EXTRINSICS)
                    else:
                        pred_values_drawable = pred_values

                    all_actions = info["actions"]

                    pred_values_drawable = cam_frame_to_cam_pixels(
                        pred_values_drawable, CURR_INTRINSICS
                    )

                    im = np.array(im, dtype="uint8")
                    frame = draw_dot_on_frame(
                        im, pred_values_drawable[[0, 10, 20, 30, 40, 50, 60, 70, 80, 90]], show=False, palette="Greens"
                    )


                    # Draw ee_pose
                    ee_pose_input = aloha_fk.fk(qpos[:, 7:13]).to(device)
                    ee_pose_cam_frame = ee_pose_to_cam_frame(
                        ee_pose_input.cpu().numpy(), CURR_EXTRINSICS
                    )[:, None, :]
                    ee_pose_pixels = cam_frame_to_cam_pixels(
                        ee_pose_cam_frame[0], CURR_INTRINSICS
                    )
                    frame = draw_dot_on_frame(
                        frame, ee_pose_pixels, show=False, palette="Set1"
                    )


                    # Save images
                    rollout_images.append(frame)
                    plt.imsave(os.path.join(rollout_dir, f"viz{t}.png"), frame)
                    plt.imsave(os.path.join(rollout_dir, f"wrist_rgb{t}.png"), data["obs"]["right_wrist_img"][0, 0].cpu().numpy())

                    if False:
                        all_actions_numpy = all_actions.cpu().numpy()
                        # fig, ax = plt.subplots()
                        # ax = plot_joint_pos(ax, all_actions_numpy)
                        all_actions_numpy = scipy.ndimage.gaussian_filter1d(
                            all_actions_numpy, axis=1, sigma=2
                        )
                        # ax = plot_joint_pos(ax, all_actions_numpy, linestyle="dotted")
                        # fig.savefig(os.path.join(ckpt_dir, f"rolloutViz_{rollout_id}", "actions.png"))
                        all_actions = torch.from_numpy(all_actions_numpy).to(
                            all_actions.device
                        )

                raw_action = all_actions[:, t % query_frequency]

                ### post-process actions
                raw_action = raw_action.squeeze(0).cpu().numpy()
                # action = post_process(raw_action)
                target_qpos = raw_action
                # target_qpos = action

                ### step the environment
                target_qpos = np.concatenate([np.zeros(7), target_qpos])
                ts = env.step(target_qpos)
                time.sleep(DT*2)


        # save_images(rollout_images, viz_dir)
        # write_vid(rollout_images, os.path.join(viz_dir, "video_0.mp4"))
        rollout_images = []

        print("moving robot")
        move_grippers(
            [env.puppet_bot_right], [PUPPET_GRIPPER_JOINT_OPEN] * 2, move_time=0.5
        )  # open
    return


def main(args):
    """
    Train a model using the algorithm.
    """

    ext_cfg = json.load(open(args.config, "r"))
    config = config_factory(ext_cfg["algo_name"])
    # update config with external json - this will throw errors if
    # the external config has keys not present in the base algo config
    with config.values_unlocked():
        config.update(ext_cfg)

    # first set seeds
    np.random.seed(101)
    torch.manual_seed(101)

    # print("\n============= New Training Run with Config =============")
    # print(config)
    # print("")
    # log_dir, ckpt_dir, video_dir, uid = get_exp_dir(config)

    # breakpoint()
    model = ModelWrapper.load_from_checkpoint(args.eval_path, datamodule=None)
    
    node = create_interbotix_global_node('aloha')
    env = make_real_env(node, active_arms="right", setup_robots=False)
    robot_startup(node)
    model.eval()
    rollout_dir = os.path.dirname(os.path.dirname(args.eval_path))
    rollout_dir = os.path.join(rollout_dir, "rollouts")
    if not os.path.exists(rollout_dir):
        os.mkdir(rollout_dir)

    eval_real(model.model, config, env, rollout_dir)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # External config file that overwrites default config
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="(optional) path to a config json that will be used to override the default settings. \
            If omitted, default settings are used. This is the preferred way to run experiments.",
    )

    # Algorithm Name
    # parser.add_argument(
    #     "--algo",
    #     type=str,
    #     help="(optional) name of algorithm to run. Only needs to be provided if --config is not provided",
    # )

    # Experiment Name (for tensorboard, saving models, etc.)
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="(optional) if provided, override the experiment name defined in the config",
    )

    parser.add_argument(
        "--description",
        type=str,
        default=None,
        help="(optional) if provided, override the experiment description defined in the config",
    )

    parser.add_argument(
        "--eval-path",
        type=str,
        default=None,
        help="(optional) path to the model to be evaluated",
    )

    parser.add_argument(
        "--gen-vid",
        type=int,
        default=1,
        help="(optional) whether to generate videos or not 0 false 1 true.",
        choices=[0, 1],
    )

    args = parser.parse_args()
    # if "DT" not in args.description:
    #     time_str = f"{args.description}_DT_{datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M-%S')}"
    #     args.description = time_str
    main(args)
