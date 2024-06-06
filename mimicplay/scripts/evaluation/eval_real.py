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
from mimicplay.utils.file_utils import policy_from_checkpoint
from torchvision.utils import save_image
import cv2

PUPPET_GRIPPER_JOINT_OPEN=1.4910

from mimicplay.scripts.aloha_process.simarUtils import cam_frame_to_cam_pixels, draw_dot_on_frame, general_unnorm, miniviewer, nds, WIDE_LENS_ROBOT_LEFT_K, EXTRINSICS, ee_pose_to_cam_frame, AlohaFK
import torchvision


from mimicplay.configs import config_factory
from mimicplay.algo import algo_factory, RolloutPolicy
from mimicplay.utils.train_utils import get_exp_dir, rollout_with_stats, load_data_for_training
from mimicplay.utils.val_utils import evaluate_high_level_policy
from mimicplay.scripts.pl_train import ModelWrapper
import datetime

from aloha_scripts.robot_utils import move_grippers # requires aloha
from aloha_scripts.real_env import make_real_env # requires aloha

from mimicplay.scripts.evaluation.real_utils import *
import matplotlib.pyplot as plt
from mimicplay.algo.act import ACT



CURR_INTRINSICS = WIDE_LENS_ROBOT_LEFT_K
CURR_EXTRINSICS = EXTRINSICS["humanoidApr16"]

def eval_real(model, config, env, rollout_dir):
    real_robot = True
    device = torch.device("cuda")
    #TODO get camnames from config
    camera_names = ['cam_high', 'cam_right_wrist']
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

        ### onscreen render
        # if onscreen_render:
        #     ax = plt.subplot(1, 1, 1)
        #     plt_img = ax.imshow(np.zeros((240, 320, 3)).astype(np.uint8))
        #     # TODO: put current image, goal image, wrist image, + predictions on the subplots during rollout.  Make this a separate function
        #     plt.ion()

        ### evaluation loop
        # if temporal_agg:
        #     all_time_actions = torch.zeros([max_timesteps, max_timesteps + num_queries, state_dim]).cuda()

        # qpos_history = torch.zeros((1, max_timesteps, state_dim)).to(device)
        image_list = []  # for visualization
        qpos_list = []
        target_qpos_list = []
        rewards = []

        t0 = time.time()
        with torch.inference_mode():
            rollout_images = []
            for t in range(700):

                ### process previous timestep to get qpos and image_list
                obs = ts.observation
                if 'images' in obs:
                    image_list.append(obs['images'])
                else:
                    image_list.append({'main': obs['image']})
                qpos_numpy = np.array(obs['qpos'])
                # qpos = pre_process(qpos_numpy)
                qpos = qpos_numpy
                qpos = torch.from_numpy(qpos).float().unsqueeze(0).to(device)
                # qpos_history[:, t] = qpos
                curr_image = get_image(ts, camera_names, device)
                # curr_image = resize_curr_img(curr_image)

                ### query policy
                if t % query_frequency == 0:
                    ee_pose_input = aloha_fk.fk(qpos[:, 7:13]).to(device)
                    cv2.imwrite(os.path.join(rollout_dir, "wrist_rgb.png"), curr_image[:, [1]][0][0].permute(1, 2, 0).cpu().numpy()*255.0)
                    ee_pose_cam_frame= ee_pose_to_cam_frame(ee_pose_input.cpu().numpy(), CURR_EXTRINSICS)[:, None, :]
                    ee_pose_pixels = cam_frame_to_cam_pixels(ee_pose_cam_frame[0], CURR_INTRINSICS)
                    data = {
                        "obs": {
                            "front_img_1": (curr_image[:, [0]].permute((0, 1, 3, 4, 2))*255).to(torch.uint8),
                            "right_wrist_img": (curr_image[:, [1]].permute((0, 1, 3, 4, 2))*255).to(torch.uint8),
                            "ee_pose": torch.from_numpy(ee_pose_cam_frame), #torch.tensor([[[0.2889, 0.1556, 0.4028]]]), #ee_pose_input[:, None, :], #TODO: Switch this to actual qpos (and make corresponding change in config)
                            "pad_mask": torch.ones((1, 100, 1)).to(device).bool()
                        }
                    }

                    input_batch = model.process_batch_for_training(data)
                    input_batch = model.postprocess_batch_for_training(input_batch, obs_normalization_stats=None) # TODO: look into obs norm
                    # if GOAL_COND and "ee_pose" in input_batch["goal_obs"]:
                    #     del input_batch["goal_obs"]["ee_pose"]
                    # del input_batch["actions"]
                    info = model.forward_eval(input_batch)

                    ##### DHRUV ######
                    im = data["obs"]["front_img_1"][0, 0].cpu().numpy()
                    if isinstance(model, ACT):
                        pred_values = info["actions"][0].cpu().numpy()
                        actions = info["actions"][0].cpu().numpy()
                        # actions = input_batch["actions"][0].cpu().numpy()
                    else:
                        pred_values = info.mean[0].view((10,3)).cpu().numpy()
                        actions = input_batch["actions"][0, 0].view((10, 3)).cpu().numpy()

                    if model.ac_key == "actions_joints":
                        pred_values_drawable, actions_drawable = aloha_fk.fk(pred_values[:, :6]), aloha_fk.fk(actions[:, :6])
                        pred_values_drawable, actions_drawable = ee_pose_to_cam_frame(pred_values_drawable, CURR_EXTRINSICS), ee_pose_to_cam_frame(actions_drawable, CURR_EXTRINSICS)
                        actions_base_frame = aloha_fk.fk(actions[:, :6])

                        actions_pixels = ee_pose_to_cam_pixels(actions_base_frame, CURR_EXTRINSICS, CURR_INTRINSICS/2)/2
                    else:
                        pred_values_drawable, actions_drawable = pred_values, actions


                    all_actions = info["actions"]
                    # breakpoint()
                    pred_values_drawable = cam_frame_to_cam_pixels(pred_values_drawable, CURR_INTRINSICS)
                    actions_drawable = cam_frame_to_cam_pixels(actions_drawable, CURR_INTRINSICS)
                    # frame = draw_dot_on_frame(im, actions_pixels, show=False, palette="Purples")
                    im = np.array(im, dtype='uint8')
                    frame = draw_dot_on_frame(im, actions_drawable, show=False, palette="Greens")
                    frame =  draw_dot_on_frame(frame, ee_pose_pixels, show=False, palette="Set1")

                    # plt.imshow(frame)
                    # plt.show()
                    rollout_images.append(frame)
                    # viz_dir = os.path.join(rollout_dir, f"rolloutViz_{rollout_id}")
                    plt.imsave(os.path.join(rollout_dir, "viz.png"), frame)

                    if True:
                        all_actions_numpy = all_actions.cpu().numpy()
                        # fig, ax = plt.subplots()
                        # ax = plot_joint_pos(ax, all_actions_numpy)
                        all_actions_numpy = scipy.ndimage.gaussian_filter1d(all_actions_numpy, axis=1, sigma=2)
                        # ax = plot_joint_pos(ax, all_actions_numpy, linestyle="dotted")
                        # fig.savefig(os.path.join(ckpt_dir, f"rolloutViz_{rollout_id}", "actions.png"))
                        all_actions = torch.from_numpy(all_actions_numpy).to(all_actions.device)

                raw_action = all_actions[:, t % query_frequency]

                ### post-process actions
                raw_action = raw_action.squeeze(0).cpu().numpy()
                # action = post_process(raw_action)
                target_qpos = raw_action
                # target_qpos = action

                ### step the environment
                target_qpos = np.concatenate([np.zeros(7), target_qpos])
                ts = env.step(target_qpos)

                ### for visualization
                qpos_list.append(qpos_numpy)
                target_qpos_list.append(target_qpos)
                #####       ######

        # save_images(rollout_images, viz_dir)
        # write_vid(rollout_images, os.path.join(viz_dir, "video_0.mp4"))
        rollout_images = []
        if real_robot:
            print("moving robot")
            move_grippers([env.puppet_bot_right], [PUPPET_GRIPPER_JOINT_OPEN] * 2,
                          move_time=0.5)  # open
            pass
    return

def init_robomimic_alg(config):
    ObsUtils.initialize_obs_utils_with_config(config)

    # print("\n============= Loaded Environment Metadata =============")
    # dataset_path = config.train.data
    # shape_meta = FileUtils.get_shape_metadata_from_dataset(
    #     dataset_path=dataset_path,
    #     all_obs_keys=config.all_obs_keys,
    #     verbose=True,
    #     ac_key=config.train.ac_key
    # )

    shape_meta = {'ac_dim': 7, 'all_shapes': OrderedDict([('ee_pose', [3]), ('front_img_1', [3, 480, 640]), ('right_wrist_img', [3, 480, 640])]), 'all_obs_keys': ['ee_pose', 'front_img_1', 'right_wrist_img'], 'use_images': True, 'use_depths': False} #TODO: this is hardcoded, and would break when we switch ee_pose to actions, but it's an easy fix

    # setup for a new training runs
    model = algo_factory(
        algo_name=config.algo_name,
        config=config,
        obs_key_shapes=shape_meta["all_shapes"],
        ac_dim=shape_meta["ac_dim"],
        device="cuda"  # default to cpu, pl will move to gpu
    )

    return model


def main(args):
    """
    Train a model using the algorithm.
    """

    ext_cfg = json.load(open(args.config, 'r'))
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

    if args.dataset is not None:
        config.train.data = args.dataset

    # breakpoint()
    model = ModelWrapper.load_from_checkpoint(args.eval_path, datamodule=None)
    
    env = make_real_env(init_node=True, arm_left=False, arm_right=True)
    model.eval()
    rollout_dir = os.path.dirname(os.path.dirname(args.eval_path))
    rollout_dir = os.path.join(rollout_dir, "rollouts")
    if not os.path.exists(rollout_dir):
        os.mkdir(rollout_dir)

    eval_real(model.model, config, env, rollout_dir)


# def main(args):
#     # if args.config is not None:
#     #     ext_cfg = json.load(open(args.config, 'r'))
#     #     config = config_factory(ext_cfg["algo_name"])
#     #     # update config with external json - this will throw errors if
#     #     # the external config has keys not present in the base algo config
#     #     with config.values_unlocked():
#     #         config.update(ext_cfg)
#     # else:
#     #     config = config_factory(args.algo)

#     # if args.dataset is not None:
#     #     config.train.data = args.dataset

#     # if args.name is not None:
#     #     config.experiment.name = args.name

#     # if args.name is not None:
#     #     config.experiment.description = args.description

#     # get torch device
#     device = TorchUtils.get_torch_device(try_to_use_cuda=config.train.cuda)

#     # lock config to prevent further modifications and ensure missing keys raise errors
#     config.lock()

#     # catch error during training and print it
#     res_str = "finished run successfully!"
#     try:
#         train(config, device=device)
#     except Exception as e:
#         res_str = "run failed with error:\n{}\n\n{}".format(e, traceback.format_exc())
#     print(res_str)


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

    # Dataset path, to override the one in the config
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="(optional) if provided, override the dataset path defined in the config",
    )

    parser.add_argument("--eval-path", type=str, default=None, help="(optional) path to the model to be evaluated")

    parser.add_argument("--gen-vid", type=int, default=1, help="(optional) whether to generate videos or not 0 false 1 true.", choices=[0, 1])

    args = parser.parse_args()
    # if "DT" not in args.description:
    #     time_str = f"{args.description}_DT_{datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M-%S')}"
    #     args.description = time_str
    main(args)

