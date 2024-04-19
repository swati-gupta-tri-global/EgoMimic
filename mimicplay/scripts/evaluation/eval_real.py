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

from mimicplay.scripts.aloha_process.simarUtils import cam_frame_to_cam_pixels, draw_dot_on_frame, general_unnorm, miniviewer, nds
import torchvision


from mimicplay.configs import config_factory
from mimicplay.algo import algo_factory, RolloutPolicy
from mimicplay.utils.train_utils import get_exp_dir, rollout_with_stats, load_data_for_training
from mimicplay.utils.val_utils import evaluate_high_level_policy
import datetime

from aloha_scripts.robot_utils import move_grippers # requires aloha
from aloha_scripts.real_env import make_real_env # requires aloha

from mimicplay.scripts.evaluation.real_utils import *

def eval_real(model, config, env, device):
    #TODO get camnames from config
    camera_names = ['cam_high', 'cam_right_wrist']

    # query_frequency = policy_config['num_queries']
    # if temporal_agg:
    #     query_frequency = 1
    #     num_queries = policy_config['num_queries']
    query_frequency = 100

    max_timesteps = int(max_timesteps * 1)  # may increase for real-world tasks

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
                curr_image = get_image(ts, camera_names)
                curr_image = resize_curr_img(curr_image)

                ### query policy
                if config['policy_class'] == "ACT":
                    if t % query_frequency == 0:
                        # target_frame = int((time.time() - t0) * (1/DT) + 150) # linear extrap

                        # _, target_frame = kd_tree.query(qpos_numpy[7:], 40)
                        # print("possible target frames: ", target_frame)
                        # target_frame = np.min(target_frame)
                        
                        # target_frame = min(699, target_frame + 150) #set goal to 150 frames ahead of closest match
                        # print("target frame: ", target_frame)
                        # target_frame = 200


                        # goal_im = torch.from_numpy(prompt_data["observations/images/cam_high"][target_frame]).float().permute((2, 0, 1))[None, None, :] / 255
                        # goal_im = resize_curr_img(goal_im)
                        all_actions, forward_dict = model(qpos, curr_image, goal_im)

                        
                        if True:
                            all_actions_numpy = all_actions.cpu().numpy()
                            # fig, ax = plt.subplots()
                            # ax = plot_joint_pos(ax, all_actions_numpy)
                            all_actions_numpy = scipy.ndimage.gaussian_filter1d(all_actions_numpy, axis=1, sigma=2)
                            # ax = plot_joint_pos(ax, all_actions_numpy, linestyle="dotted")
                            # fig.savefig(os.path.join(ckpt_dir, f"rolloutViz_{rollout_id}", "actions.png"))
                            all_actions = torch.from_numpy(all_actions_numpy).to(all_actions.device)
                    if t % query_frequency == 0:
                        rend_imgs, _ = render_trajs_batch(
                            img_data=curr_image.cpu().numpy(),
                            traj_dict=forward_dict,
                            cam2base=CURR_EXTRINSICS,
                            K=CURR_INTRINSICS,
                            colors={
                                "guide_traj_base_frame": "Reds",
                                "right_pred_eef_base_frame": "Purples", 
                                "r_ee_pos": "Greens"
                            }
                        )
                        
                        
                        rend_imgs[0] = miniviewer(rend_imgs[0], (255 * goal_im[0, 0].cpu().numpy().transpose((1, 2, 0))).astype(np.uint8))
                        rend_imgs[0] = miniviewer(rend_imgs[0], (255*curr_image[0, 1].cpu().numpy().transpose(1, 2, 0)).astype(np.uint8), location="top_left")
                        if not onscreen_render:
                            rollout_images += rend_imgs
                        viz_dir = os.path.join(ckpt_dir, f"rolloutViz_{rollout_id}")
                        # breakpoint()

                        ### update onscreen render and wait for DT
                        if onscreen_render:
                            plt_img.set_data(rend_imgs[0])
                            plt.pause(DT)
                    
                    if temporal_agg:
                        all_time_actions[[t], t:t + num_queries] = all_actions
                        actions_for_curr_step = all_time_actions[:, t]
                        actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                        actions_for_curr_step = actions_for_curr_step[actions_populated]
                        k = 0.01
                        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                        exp_weights = exp_weights / exp_weights.sum()
                        exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                        raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                    else:
                        raw_action = all_actions[:, t % query_frequency]
                elif config['policy_class'] == "CNNMLP":
                    raw_action = policy(qpos, curr_image,)
                else:
                    raise NotImplementedError

                ### post-process actions
                raw_action = raw_action.squeeze(0).cpu().numpy()
                # action = post_process(raw_action)
                target_qpos = raw_action
                # target_qpos = action

                ### step the environment
                ts = env.step(target_qpos)

                ### for visualization
                qpos_list.append(qpos_numpy)
                target_qpos_list.append(target_qpos)
                rewards.append(ts.reward)

                # if real_robot and 

            plt.close()
        save_images(rollout_images, viz_dir)
        # write_vid(rollout_images, os.path.join(viz_dir, "video_0.mp4"))
        rollout_images = []
        if real_robot:
            print("moving robot")
            move_grippers([env.puppet_bot_left, env.puppet_bot_right], [PUPPET_GRIPPER_JOINT_OPEN] * 2,
                          move_time=0.5)  # open
            pass

    return



def train(config, device):
    """
    Train a model using the algorithm.
    """

    # first set seeds
    np.random.seed(config.train.seed)
    torch.manual_seed(config.train.seed)

    print("\n============= New Training Run with Config =============")
    print(config)
    print("")
    log_dir, ckpt_dir, video_dir, uid = get_exp_dir(config)

    if config.experiment.logging.terminal_output_to_txt:
        # log stdout and stderr to a text file
        logger = PrintLogger(os.path.join(log_dir, 'log.txt'))
        sys.stdout = logger
        sys.stderr = logger

    # read config to set up metadata for observation modalities (e.g. detecting rgb observations)
    ObsUtils.initialize_obs_utils_with_config(config)

    # make sure the dataset exists
    dataset_path = os.path.expanduser(config.train.data)
    if not os.path.exists(dataset_path):
        raise Exception("Dataset at provided path {} not found!".format(dataset_path))

    # load basic metadata from training file
    print("\n============= Loaded Environment Metadata =============")
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=config.train.data)
    shape_meta = FileUtils.get_shape_metadata_from_dataset(
        dataset_path=config.train.data,
        all_obs_keys=config.all_obs_keys,
        verbose=True
    )

    model = policy_from_checkpoint(device=device, ckpt_path=args.eval_path, ckpt_dict=None, verbose=False)


    print("\n============= Model Summary =============")
    print(model[0].policy)  # print model summary
    print("")

    trainset, validset = load_data_for_training(
        config, obs_keys=shape_meta["all_obs_keys"])
    train_sampler = trainset.get_dataset_sampler()
    print("\n============= Training Dataset =============")
    print(trainset)
    print("")

    # maybe retreve statistics for normalizing observations
    obs_normalization_stats = None
    if config.train.hdf5_normalize_obs:
        obs_normalization_stats = trainset.get_obs_normalization_stats()

    if "obs_mins" not in env_meta:
        obs_mins = None
        obs_maxs = None
    else:
        obs_mins = np.array(env_meta["obs_mins"])
        obs_maxs = np.array(env_meta["obs_maxs"])

    if not args.gen_vid:
        video_dir=None
    
    env = make_real_env(init_node=True, arm_left=False, arm_right=True)
    model.eval()

    eval_real(model, env)


def main(args):
    if args.config is not None:
        ext_cfg = json.load(open(args.config, 'r'))
        config = config_factory(ext_cfg["algo_name"])
        # update config with external json - this will throw errors if
        # the external config has keys not present in the base algo config
        with config.values_unlocked():
            config.update(ext_cfg)
    else:
        config = config_factory(args.algo)

    if args.dataset is not None:
        config.train.data = args.dataset

    if args.name is not None:
        config.experiment.name = args.name

    if args.name is not None:
        config.experiment.description = args.description

    # get torch device
    device = TorchUtils.get_torch_device(try_to_use_cuda=config.train.cuda)

    # lock config to prevent further modifications and ensure missing keys raise errors
    config.lock()

    # catch error during training and print it
    res_str = "finished run successfully!"
    try:
        train(config, device=device)
    except Exception as e:
        res_str = "run failed with error:\n{}\n\n{}".format(e, traceback.format_exc())
    print(res_str)


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
    if "DT" not in args.description:
        time_str = f"{args.description}_DT_{datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M-%S')}"
        args.description = time_str
    main(args)

