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

    if config.experiment.env is not None:
        env_meta["env_name"] = config.experiment.env
        print("=" * 30 + "\n" + "Replacing Env to {}\n".format(env_meta["env_name"]) + "=" * 30)

    # create environment
    envs = OrderedDict()
    if config.experiment.rollout.enabled:
        # create environments for validation runs
        env_names = [env_meta["env_name"]]

        if config.experiment.additional_envs is not None:
            for name in config.experiment.additional_envs:
                env_names.append(name)

        for env_name in env_names:
            dummy_spec = dict(
                obs=dict(
                    low_dim=config.observation.modalities.obs.low_dim,
                    rgb=config.observation.modalities.obs.rgb,
                ),
            )
            ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs=dummy_spec)

            if args.bddl_file is not None:
                env_meta["env_kwargs"]['bddl_file_name'] = args.bddl_file

            print(env_meta)

            env = EnvUtils.create_env_from_metadata(
                env_meta=env_meta,
                env_name=env_name,
                render=False,
                render_offscreen=config.experiment.render_video,
                use_image_obs=shape_meta["use_images"],
            )
            envs[env.name] = env

    # setup for a new training run
    data_logger = DataLogger(
        log_dir,
        config,
        uid=uid,
        log_tb=config.experiment.logging.log_tb,
    )
    # model = algo_factory(
    #     algo_name=config.algo_name,
    #     config=config,
    #     obs_key_shapes=shape_meta["all_shapes"],
    #     ac_dim=shape_meta["ac_dim"],
    #     device=device,
    # )
    model = policy_from_checkpoint(device=device, ckpt_path=args.eval_path, ckpt_dict=None, verbose=False)

    if config.experiment.rollout.enabled:                     # load task video prompt (used for evaluation rollouts during the gap of training)
        model.load_eval_video_prompt(args.video_prompt)

    # save the config as a json file
    with open(os.path.join(log_dir, '..', 'config.json'), 'w') as outfile:
        json.dump(config, outfile, indent=4)

    print("\n============= Model Summary =============")
    print(model[0].policy)  # print model summary
    print("")

    # load training data
    # FIXED_GOAL_RANGE = [150, 150]
    # config["algo"]["playdata"]["goal_image_range"] = FIXED_GOAL_RANGE
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

    # initialize data loaders
    train_loader = DataLoader(
        dataset=trainset,
        # sampler=train_sampler,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=config.train.num_data_workers,
        # num_workers=0,
        drop_last=True
    )

    if config.experiment.validate:
        # cap num workers for validation dataset at 1
        num_workers = min(config.train.num_data_workers, 1)
        valid_sampler = validset.get_dataset_sampler()
        valid_loader = DataLoader(
            dataset=validset,
            sampler=valid_sampler,
            batch_size=config.train.batch_size,
            shuffle=False,
            num_workers=num_workers,
            # num_workers=0,
            drop_last=True
        )
    else:
        valid_loader = None

    # model.load_state_dict(torch.load(args.eval_path))

    loader = valid_loader if args.eval_split == "valid" else train_loader

    if "obs_mins" not in env_meta:
        obs_mins = None
        obs_maxs = None
    else:
        obs_mins = np.array(env_meta["obs_mins"])
        obs_maxs = np.array(env_meta["obs_maxs"])

    if not args.gen_vid:
        video_dir=None
    evaluate_high_level_policy(model[0].policy, loader, video_dir=video_dir)

    # terminate logging
    data_logger.close()

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

    # Dataset path, to override the one in the config
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="(optional) if provided, override the dataset path defined in the config",
    )

    parser.add_argument("--eval-path", type=str, default=None, help="(optional) path to the model to be evaluated")

    parser.add_argument("--eval-split", type=str, default="valid", help="(optional) split to evaluate on", choices=["train", "valid"])

    parser.add_argument("--gen-vid", type=int, default=1, help="(optional) whether to generate videos or not 0 false 1 true.", choices=[0, 1])
    # parser.add_argument(
    #     "--bddl_file",
    #     type=str,
    #     default=None,
    #     help="(optional) if provided, the task's goal is specified as the symbolic goal in the bddl file (several symbolic predicates connected with AND / OR)",
    # )

    # parser.add_argument(
    #     "--video_prompt",
    #     type=str,
    #     default=None,
    #     help="(optional) if provided, a task video prompt is loaded and used in the evaluation rollouts",
    # )

    # debug mode
    # parser.add_argument(
    #     "--debug",
    #     action='store_true',
    #     help="set this flag to run a quick training run for debugging purposes"
    # )

    args = parser.parse_args()
    main(args)

