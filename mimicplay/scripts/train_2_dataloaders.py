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
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
import heapq
import h5py

import datetime
from collections import OrderedDict

import torch
from torch.utils.data import DataLoader
import torch.distributed as distrib

import robomimic.utils.train_utils as TrainUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
from robomimic.utils.log_utils import PrintLogger, DataLogger

from mimicplay.configs import config_factory
from mimicplay.algo import algo_factory, RolloutPolicy
from mimicplay.utils.train_utils import get_exp_dir, rollout_with_stats, load_data_for_training

import mimicplay.utils.val_utils as ValUtils
from mimicplay.scripts.ddp_utils import convert_groupnorm_model, init_distrib_slurm, EXIT

def get_gpu_usage_mb(index=0):
    """Returns the GPU usage in B."""
    h = nvmlDeviceGetHandleByIndex(index)
    info = nvmlDeviceGetMemoryInfo(h)
    print(f'total    : {info.total}')
    print(f'free     : {info.free}')
    print(f'used     : {info.used}')

    return info.used / 1024 / 1024

def train_ddp(config):
    """
    Train a model on multiple GPUs using the algorithm.
    """
    nvmlInit()
    local_rank, _ = init_distrib_slurm(backend="gloo")
    world_rank = distrib.get_rank()
    world_size = distrib.get_world_size()

    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    
    distrib.barrier()
    # first set seeds
    np.random.seed(config.train.seed)
    torch.manual_seed(config.train.seed)

    print("\n============= New Training Run with Config =============")
    print(config)
    print("")

    log_dir, ckpt_dir, video_dir, uid = get_exp_dir(config)
    assert config.experiment.save.video_freq >= config.experiment.validation_freq, "video_freq must be less than validation_freq"
    assert config.experiment.save.video_freq % config.experiment.validation_freq == 0, "video_freq must be a multiple of validation_freq"
    # if config.experiment.logging.terminal_output_to_txt:
    #     # log stdout and stderr to a text file
    #     logger = PrintLogger(os.path.join(log_dir, 'log.txt'))
    #     sys.stdout = logger
    #     sys.stderr = logger

    # read config to set up metadata for observation modalities (e.g. detecting rgb observations)
    ObsUtils.initialize_obs_utils_with_config(config)

    # make sure the dataset exists
    dataset_path = os.path.expanduser(config.train.data)
    if not os.path.exists(dataset_path):
        raise Exception("Dataset at provided path {} not found!".format(dataset_path))

    # load basic metadata from training file
    print("\n============= Loaded Environment Metadata =============")
    h5py_file = h5py.File(dataset_path, "r+")
    if h5py_file["data"].get("env_args") == None:
        h5py_file["data"].attrs["env_args"] = json.dumps({})
        print("Added empty env_args")
    
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
        uid=f"{config.experiment.name}_{uid}",
        # log_tb=config.experiment.logging.log_tb,
        log_wandb=config.experiment.logging.log_wandb,
    )
    model = algo_factory(
        algo_name=config.algo_name,
        config=config,
        obs_key_shapes=shape_meta["all_shapes"],
        ac_dim=shape_meta["ac_dim"],
        device=device,
    )

    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[device], output_device=device
    )
    if config.experiment.rollout.enabled:                     # load task video prompt (used for evaluation rollouts during the gap of training)
        model.load_eval_video_prompt(args.video_prompt)

    # save the config as a json file
    with open(os.path.join(log_dir, '..', 'config.json'), 'w') as outfile:
        json.dump(config, outfile, indent=4)
    
    print("\n============= Model Summary =============")
    print(model)  # print model summary
    print("")
    
    # load training data
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
        sampler=torch.utils.data.distributed.DistributedSampler(
            dataset=trainset, num_replicas=world_size, rank=world_rank
        ),
        batch_size=config.train.batch_size,
        shuffle=(train_sampler is None),
        num_workers=config.train.num_data_workers,
        drop_last=True
    )

    if config.experiment.validate:
        # cap num workers for validation dataset at 1
        num_workers = min(config.train.num_data_workers, 1)
        valid_sampler = validset.get_dataset_sampler()
        valid_loader = DataLoader(
            dataset=validset,
            sampler=torch.utils.data.distributed.DistributedSampler(
            dataset=validset, num_replicas=world_size, rank=world_rank),
            batch_size=config.train.batch_size,
            # shuffle=(valid_sampler is None),
            shuffle=False,
            num_workers=num_workers,
            drop_last=True
        )

        # video_valid_loader = DataLoader(
        #     dataset=validset,
        #     sampler=valid_sampler,
        #     batch_size=1,
        #     shuffle=False,
        #     num_workers=1,
        #     drop_last=True
        # )
    else:
        valid_loader = None

    # main training loop
    n_best_val = []
    best_return = {k: -np.inf for k in envs} if config.experiment.rollout.enabled else None
    best_success_rate = {k: -1. for k in envs} if config.experiment.rollout.enabled else None
    last_ckpt_time = time.time()

    # number of learning steps per epoch (defaults to a full dataset pass)
    train_num_steps = config.experiment.epoch_every_n_steps
    valid_num_steps = config.experiment.validation_epoch_every_n_steps
    
    for epoch in range(1, config.train.num_epochs + 1):  # epoch numbers start at 1
        train_loader.sampler.set_epoch(epoch)
        step_log = TrainUtils.run_epoch(model=model, data_loader=train_loader, epoch=epoch, num_steps=train_num_steps)
        model.on_epoch_end(epoch)

        # setup checkpoint path
        epoch_ckpt_name = "model_epoch_{}".format(epoch)

        # check for recurring checkpoint saving conditions
        should_save_ckpt = False
        if config.experiment.save.enabled:
            time_check = (config.experiment.save.every_n_seconds is not None) and \
                         (time.time() - last_ckpt_time > config.experiment.save.every_n_seconds)
            epoch_check = (config.experiment.save.every_n_epochs is not None) and \
                          (epoch > 0) and (epoch % config.experiment.save.every_n_epochs == 0)
            epoch_list_check = (epoch in config.experiment.save.epochs)
            should_save_ckpt = (time_check or epoch_check or epoch_list_check)
        ckpt_reason = None
        if should_save_ckpt:
            last_ckpt_time = time.time()
            ckpt_reason = "time"

        print("Train Epoch {}".format(epoch))
        print(json.dumps(step_log, sort_keys=True, indent=4))
        for k, v in step_log.items():
            if k.startswith("Time_"):
                data_logger.record("Timing_Stats/Train_{}".format(k[5:]), v, epoch)
            else:
                data_logger.record("Train/{}".format(k), v, epoch)

        # Evaluate the model on validation set
        if config.experiment.validate and (epoch % config.experiment.validation_freq == 0):
            with torch.no_grad():
                step_log = TrainUtils.run_epoch(model=model, data_loader=valid_loader, epoch=epoch, validate=True,
                                                num_steps=valid_num_steps)

                model.set_eval()

                pass_vid = video_dir if config.experiment.save.video_freq is not None and epoch % config.experiment.save.video_freq == 0 else None
                valid_step_log = ValUtils.evaluate_high_level_policy(model, valid_loader, pass_vid) #save vid only once every video_freq epochs

                model.set_train()
            for k, v in step_log.items():
                if k.startswith("Time_"):
                    data_logger.record("Timing_Stats/Valid_{}".format(k[5:]), v, epoch)
                else:
                    data_logger.record("Valid/{}".format(k), v, epoch)
            for k, v in valid_step_log.items():
                data_logger.record(f"Valid/{k}", v, epoch)

            print("Validation Epoch {}".format(epoch))
            print(json.dumps(step_log, sort_keys=True, indent=4))

            # save checkpoint if achieve new best validation loss
            valid_check = "Loss" in step_log
            negator = -1
            while len(n_best_val) > config.experiment.save.top_n:
                _, to_delete = heapq.heappop(n_best_val)
                if os.path.exists(to_delete):
                    os.remove(to_delete)
                else:
                    print(f"Warning: {to_delete} does not exist")
            if len(n_best_val) < config.experiment.save.top_n or step_log["Loss"] < negator * n_best_val[0][0]:
                heapq.heappush(
                    n_best_val,
                    (negator * step_log["Loss"], os.path.join(ckpt_dir, epoch_ckpt_name + "_best_validation_{}".format(step_log["Loss"])) + ".pth")
                ) #negate to make max heap
                is_top_n = True
            else:
                is_top_n = False

            if valid_check and (n_best_val is None or is_top_n):
                if config.experiment.save.enabled and config.experiment.save.on_best_validation:
                    epoch_ckpt_name += "_best_validation_{}".format(step_log["Loss"])
                    should_save_ckpt = True
                    ckpt_reason = "valid" if ckpt_reason is None else ckpt_reason

        # Evaluate the model by running rollouts

        # do rollouts at fixed rate or if it's time to save a new ckpt
        video_paths = None
        rollout_check = (epoch % config.experiment.rollout.rate == 0) or (should_save_ckpt and ckpt_reason == "time")
        if config.experiment.rollout.enabled and (epoch > config.experiment.rollout.warmstart) and rollout_check:

            # wrap model as a RolloutPolicy to prepare for rollouts
            rollout_model = RolloutPolicy(model, obs_normalization_stats=obs_normalization_stats)

            num_episodes = config.experiment.rollout.n
            all_rollout_logs, video_paths = rollout_with_stats(
                policy=rollout_model,
                envs=envs,
                horizon=config.experiment.rollout.horizon,
                use_goals=config.use_goals,
                num_episodes=num_episodes,
                render=False,
                video_dir=video_dir if config.experiment.render_video else None,
                epoch=epoch,
                video_skip=config.experiment.get("video_skip", 5),
                terminate_on_success=config.experiment.rollout.terminate_on_success,
            )

            # summarize results from rollouts to tensorboard and terminal
            for env_name in all_rollout_logs:
                rollout_logs = all_rollout_logs[env_name]
                for k, v in rollout_logs.items():
                    if k.startswith("Time_"):
                        data_logger.record("Timing_Stats/Rollout_{}_{}".format(env_name, k[5:]), v, epoch)
                    else:
                        data_logger.record("Rollout/{}/{}".format(k, env_name), v, epoch, log_stats=True)

                print("\nEpoch {} Rollouts took {}s (avg) with results:".format(epoch, rollout_logs["time"]))
                print('Env: {}'.format(env_name))
                print(json.dumps(rollout_logs, sort_keys=True, indent=4))

            # checkpoint and video saving logic
            updated_stats = TrainUtils.should_save_from_rollout_logs(
                all_rollout_logs=all_rollout_logs,
                best_return=best_return,
                best_success_rate=best_success_rate,
                epoch_ckpt_name=epoch_ckpt_name,
                save_on_best_rollout_return=config.experiment.save.on_best_rollout_return,
                save_on_best_rollout_success_rate=config.experiment.save.on_best_rollout_success_rate,
            )
            best_return = updated_stats["best_return"]
            best_success_rate = updated_stats["best_success_rate"]
            epoch_ckpt_name = updated_stats["epoch_ckpt_name"]
            should_save_ckpt = (config.experiment.save.enabled and updated_stats[
                "should_save_ckpt"]) or should_save_ckpt
            if updated_stats["ckpt_reason"] is not None:
                ckpt_reason = updated_stats["ckpt_reason"]

        # Only keep saved videos if the ckpt should be saved (but not because of validation score)
        should_save_video = (should_save_ckpt and (ckpt_reason != "valid")) or config.experiment.keep_all_videos
        if video_paths is not None and not should_save_video:
            for env_name in video_paths:
                os.remove(video_paths[env_name])

        # Save model checkpoints based on conditions (success rate, validation loss, etc)
        if should_save_ckpt:
            TrainUtils.save_model(
                model=model,
                config=config,
                env_meta=env_meta,
                shape_meta=shape_meta,
                ckpt_path=os.path.join(ckpt_dir, epoch_ckpt_name + ".pth"),
                obs_normalization_stats=obs_normalization_stats,
            )
        
        # # Delete all checkpoints except the top 5, or checkpoints saved based on regular time intervals
        # if config.experiment.save.top_n is not None:
        #     TrainUtils.delete_checkpoints(
        #         ckpt_dir=ckpt_dir, 
        #         top_n=config.experiment.save.top_n,
        #         smallest=True) # keep the lowest val loss, change to False if using an increasing val metric


        # Finally, log memory usage in MB
        process = psutil.Process(os.getpid())
        mem_usage = int(process.memory_info().rss / 1000000)
        data_logger.record("System/RAM Usage (MB)", mem_usage, epoch)
        print("\nEpoch {} RAM Memory Usage: {} MB\n".format(epoch, mem_usage))

        # print the gpu memory usage using nvidia smi
        # print(f"\n Epoch {epoch} GPU Memory Usage: {get_gpu_usage_mb()} MB\n")

        if EXIT.is_set():
            return
    # terminate logging
    data_logger.close()

def train(config, device):
    """
    Train a model using the algorithm.
    """
    nvmlInit()

    # first set seeds
    np.random.seed(config.train.seed)
    torch.manual_seed(config.train.seed)

    print("\n============= New Training Run with Config =============")
    print(config)
    print("")
    log_dir, ckpt_dir, video_dir, uid = get_exp_dir(config)
    assert config.experiment.save.video_freq >= config.experiment.validation_freq, "video_freq must be less than validation_freq"
    assert config.experiment.save.video_freq % config.experiment.validation_freq == 0, "video_freq must be a multiple of validation_freq"
    # if config.experiment.logging.terminal_output_to_txt:
    #     # log stdout and stderr to a text file
    #     logger = PrintLogger(os.path.join(log_dir, 'log.txt'))
    #     sys.stdout = logger
    #     sys.stderr = logger

    # read config to set up metadata for observation modalities (e.g. detecting rgb observations)
    ObsUtils.initialize_obs_utils_with_config(config)

    # make sure the dataset exists
    dataset_path = os.path.expanduser(config.train.data)
    if not os.path.exists(dataset_path):
        raise Exception("Dataset at provided path {} not found!".format(dataset_path))
    
    dataset_path_2 = os.path.expanduser(config.train.data_2)
    if not os.path.exists(dataset_path_2):
        raise Exception("Dataset_2 at provided path {} not found!".format(dataset_path_2))

    # load basic metadata from training file
    print("\n============= Loaded Environment Metadata =============")
    print("DATASET_PATH -1 ", dataset_path)
    print("DATASET_PATH -2 ", dataset_path_2)

    h5py_file = h5py.File(dataset_path, "r+")
    if h5py_file["data"].get("env_args") == None:
        h5py_file["data"].attrs["env_args"] = json.dumps({})
        print("Added empty env_args")
    
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
        uid=f"{config.experiment.name}_{uid}",
        # log_tb=config.experiment.logging.log_tb,
        log_wandb=config.experiment.logging.log_wandb,
    )
    model = algo_factory(
        algo_name=config.algo_name,
        config=config,
        obs_key_shapes=shape_meta["all_shapes"],
        ac_dim=shape_meta["ac_dim"],
        device=device,
    )
    if config.experiment.rollout.enabled:                     # load task video prompt (used for evaluation rollouts during the gap of training)
        model.load_eval_video_prompt(args.video_prompt)

    # save the config as a json file
    with open(os.path.join(log_dir, '..', 'config.json'), 'w') as outfile:
        json.dump(config, outfile, indent=4)
    
    print("\n============= Model Summary =============")
    print(model)  # print model summary
    print("")
    
    # load training data
    trainset, validset = load_data_for_training(
        config, obs_keys=shape_meta["all_obs_keys"], dataset_path=dataset_path)
    trainset_2, validset_2 = load_data_for_training(
        config, obs_keys=shape_meta["all_obs_keys"], dataset_path=dataset_path_2)
    train_sampler = trainset.get_dataset_sampler()
    train_sampler_2 = trainset_2.get_dataset_sampler()
    print("\n============= Training Dataset =============")
    print(trainset)
    print(trainset_2)
    print("")

    # maybe retreve statistics for normalizing observations
    obs_normalization_stats = None
    if config.train.hdf5_normalize_obs:
        obs_normalization_stats = trainset.get_obs_normalization_stats()
        obs_normalization_stats_2 = trainset_2.get_obs_normalization_stats()
    

    ## To check which loader is robot and which is hand
    is_first_loader_hand = False
    sampler = trainset.get_dataset_sampler()
    loader = DataLoader(
        dataset=trainset,
        sampler=train_sampler,
        batch_size=config.train.batch_size,
        shuffle=(train_sampler is None),
        num_workers=config.train.num_data_workers,
        drop_last=True
    )
    iterator = iter(loader)
    sample = next(iterator)
    # breakpoint()
    if torch.all(sample['obs']['type'] == 1):
        is_first_loader_hand = True
    ## To check which loader is robot and which is hand

    if is_first_loader_hand:
        # initialize data loaders
        train_loader = DataLoader(
            dataset=trainset,
            sampler=train_sampler,
            batch_size=config.train.batch_size,
            shuffle=(train_sampler is None),
            num_workers=config.train.num_data_workers,
            drop_last=True
        )

        train_loader_2 = DataLoader(
            dataset=trainset_2,
            sampler=train_sampler_2,
            batch_size=config.train.batch_size,
            shuffle=(train_sampler_2 is None),
            num_workers=config.train.num_data_workers,
            drop_last=True
        )
    else:
        train_loader = DataLoader(
            dataset=trainset_2,
            sampler=train_sampler_2,
            batch_size=config.train.batch_size,
            shuffle=(train_sampler_2 is None),
            num_workers=config.train.num_data_workers,
            drop_last=True
        )
                
        # initialize data loaders
        train_loader_2 = DataLoader(
            dataset=trainset,
            sampler=train_sampler,
            batch_size=config.train.batch_size,
            shuffle=(train_sampler is None),
            num_workers=config.train.num_data_workers,
            drop_last=True
        )

    if config.experiment.validate:
        # cap num workers for validation dataset at 1
        num_workers = min(config.train.num_data_workers, 1)
        valid_sampler = validset.get_dataset_sampler()
        valid_sampler_2 = validset_2.get_dataset_sampler()

        if is_first_loader_hand:
            valid_loader = DataLoader(
                dataset=validset,
                sampler=valid_sampler,
                batch_size=config.train.batch_size,
                # shuffle=(valid_sampler is None),
                shuffle=False,
                num_workers=num_workers,
                drop_last=True
            )

            valid_loader_2 = DataLoader(
                dataset=validset_2,
                sampler=valid_sampler_2,
                batch_size=config.train.batch_size,
                # shuffle=(valid_sampler is None),
                shuffle=False,
                num_workers=num_workers,
                drop_last=True
            )
        else:
            valid_loader_2 = DataLoader(
                dataset=validset,
                sampler=valid_sampler,
                batch_size=config.train.batch_size,
                # shuffle=(valid_sampler is None),
                shuffle=False,
                num_workers=num_workers,
                drop_last=True
            )

            valid_loader = DataLoader(
                dataset=validset_2,
                sampler=valid_sampler_2,
                batch_size=config.train.batch_size,
                # shuffle=(valid_sampler is None),
                shuffle=False,
                num_workers=num_workers,
                drop_last=True
            )
    else:
        valid_loader = None
        valid_loader_2 =None

    # main training loop
    n_best_val = []
    best_return = {k: -np.inf for k in envs} if config.experiment.rollout.enabled else None
    best_success_rate = {k: -1. for k in envs} if config.experiment.rollout.enabled else None
    last_ckpt_time = time.time()

    # number of learning steps per epoch (defaults to a full dataset pass)
    train_num_steps = config.experiment.epoch_every_n_steps
    valid_num_steps = config.experiment.validation_epoch_every_n_steps

    for epoch in range(1, config.train.num_epochs + 1):  # epoch numbers start at 1
        step_log = TrainUtils.run_epoch_2_dataloaders(model=model, data_loader=train_loader, epoch=epoch, data_loader_2=train_loader_2, num_steps=train_num_steps)
        model.on_epoch_end(epoch)

        # setup checkpoint path
        epoch_ckpt_name = "model_epoch_{}".format(epoch)

        # check for recurring checkpoint saving conditions
        should_save_ckpt = False
        if config.experiment.save.enabled:
            time_check = (config.experiment.save.every_n_seconds is not None) and \
                         (time.time() - last_ckpt_time > config.experiment.save.every_n_seconds)
            epoch_check = (config.experiment.save.every_n_epochs is not None) and \
                          (epoch > 0) and (epoch % config.experiment.save.every_n_epochs == 0)
            epoch_list_check = (epoch in config.experiment.save.epochs)
            should_save_ckpt = (time_check or epoch_check or epoch_list_check)
        ckpt_reason = None
        if should_save_ckpt:
            last_ckpt_time = time.time()
            ckpt_reason = "time"

        print("Train Epoch {}".format(epoch))
        print(json.dumps(step_log, sort_keys=True, indent=4))
        for k, v in step_log.items():
            if k.startswith("Time_"):
                data_logger.record("Timing_Stats/Train_{}".format(k[5:]), v, epoch)
            else:
                data_logger.record("Train/{}".format(k), v, epoch)

        # Evaluate the model on validation set
        if config.experiment.validate and (epoch % config.experiment.validation_freq == 0):
            with torch.no_grad():
                # step_log = TrainUtils.run_epoch(model=model, data_loader=valid_loader, epoch=epoch, validate=True,
                #                                 num_steps=valid_num_steps)
                step_log = TrainUtils.run_epoch_2_dataloaders(model=model, data_loader=valid_loader, epoch=epoch, data_loader_2=valid_loader_2, validate=True,
                                                num_steps=valid_num_steps)
                model.set_eval()

                pass_vid = video_dir if config.experiment.save.video_freq is not None and epoch % config.experiment.save.video_freq == 0 else None
                valid_step_log = ValUtils.evaluate_high_level_policy(model, valid_loader, pass_vid, type="hand") #save vid only once every video_freq epochs
                valid_step_log_2 = ValUtils.evaluate_high_level_policy(model, valid_loader_2, pass_vid, type="robot") #save vid only once every video_freq epochs

                model.set_train()
            for k, v in step_log.items():
                if k.startswith("Time_"):
                    data_logger.record("Timing_Stats/Valid_{}".format(k[5:]), v, epoch)
                else:
                    data_logger.record("Valid/{}".format(k), v, epoch)
            for k, v in valid_step_log.items():
                data_logger.record(f"Valid/{k}", v, epoch)
            for k, v in valid_step_log_2.items():
                data_logger.record(f"Valid/{k}", v, epoch)
            
            print("Validation Epoch {}".format(epoch))
            print(json.dumps(step_log, sort_keys=True, indent=4))

            # save checkpoint if achieve new best validation loss
            valid_check = "Loss" in step_log
            negator = -1
            while len(n_best_val) > config.experiment.save.top_n:
                _, to_delete = heapq.heappop(n_best_val)
                if os.path.exists(to_delete):
                    os.remove(to_delete)
                else:
                    print(f"Warning: {to_delete} does not exist")
            if len(n_best_val) < config.experiment.save.top_n or step_log["Loss"] < negator * n_best_val[0][0]:
                heapq.heappush(
                    n_best_val,
                    (negator * step_log["Loss"], os.path.join(ckpt_dir, epoch_ckpt_name + "_best_validation_{}".format(step_log["Loss"])) + ".pth")
                ) #negate to make max heap
                is_top_n = True
            else:
                is_top_n = False

            if valid_check and (n_best_val is None or is_top_n):
                if config.experiment.save.enabled and config.experiment.save.on_best_validation:
                    epoch_ckpt_name += "_best_validation_{}".format(step_log["Loss"])
                    should_save_ckpt = True
                    ckpt_reason = "valid" if ckpt_reason is None else ckpt_reason

        # Evaluate the model by running rollouts

        # do rollouts at fixed rate or if it's time to save a new ckpt
        video_paths = None
        rollout_check = (epoch % config.experiment.rollout.rate == 0) or (should_save_ckpt and ckpt_reason == "time")
        if config.experiment.rollout.enabled and (epoch > config.experiment.rollout.warmstart) and rollout_check:

            # wrap model as a RolloutPolicy to prepare for rollouts
            rollout_model = RolloutPolicy(model, obs_normalization_stats=obs_normalization_stats)

            num_episodes = config.experiment.rollout.n
            all_rollout_logs, video_paths = rollout_with_stats(
                policy=rollout_model,
                envs=envs,
                horizon=config.experiment.rollout.horizon,
                use_goals=config.use_goals,
                num_episodes=num_episodes,
                render=False,
                video_dir=video_dir if config.experiment.render_video else None,
                epoch=epoch,
                video_skip=config.experiment.get("video_skip", 5),
                terminate_on_success=config.experiment.rollout.terminate_on_success,
            )

            # summarize results from rollouts to tensorboard and terminal
            for env_name in all_rollout_logs:
                rollout_logs = all_rollout_logs[env_name]
                for k, v in rollout_logs.items():
                    if k.startswith("Time_"):
                        data_logger.record("Timing_Stats/Rollout_{}_{}".format(env_name, k[5:]), v, epoch)
                    else:
                        data_logger.record("Rollout/{}/{}".format(k, env_name), v, epoch, log_stats=True)

                print("\nEpoch {} Rollouts took {}s (avg) with results:".format(epoch, rollout_logs["time"]))
                print('Env: {}'.format(env_name))
                print(json.dumps(rollout_logs, sort_keys=True, indent=4))

            # checkpoint and video saving logic
            updated_stats = TrainUtils.should_save_from_rollout_logs(
                all_rollout_logs=all_rollout_logs,
                best_return=best_return,
                best_success_rate=best_success_rate,
                epoch_ckpt_name=epoch_ckpt_name,
                save_on_best_rollout_return=config.experiment.save.on_best_rollout_return,
                save_on_best_rollout_success_rate=config.experiment.save.on_best_rollout_success_rate,
            )
            best_return = updated_stats["best_return"]
            best_success_rate = updated_stats["best_success_rate"]
            epoch_ckpt_name = updated_stats["epoch_ckpt_name"]
            should_save_ckpt = (config.experiment.save.enabled and updated_stats[
                "should_save_ckpt"]) or should_save_ckpt
            if updated_stats["ckpt_reason"] is not None:
                ckpt_reason = updated_stats["ckpt_reason"]

        # Only keep saved videos if the ckpt should be saved (but not because of validation score)
        should_save_video = (should_save_ckpt and (ckpt_reason != "valid")) or config.experiment.keep_all_videos
        if video_paths is not None and not should_save_video:
            for env_name in video_paths:
                os.remove(video_paths[env_name])

        # Save model checkpoints based on conditions (success rate, validation loss, etc)
        if should_save_ckpt:
            TrainUtils.save_model(
                model=model,
                config=config,
                env_meta=env_meta,
                shape_meta=shape_meta,
                ckpt_path=os.path.join(ckpt_dir, epoch_ckpt_name + ".pth"),
                obs_normalization_stats=obs_normalization_stats,
            )
        
        # # Delete all checkpoints except the top 5, or checkpoints saved based on regular time intervals
        # if config.experiment.save.top_n is not None:
        #     TrainUtils.delete_checkpoints(
        #         ckpt_dir=ckpt_dir, 
        #         top_n=config.experiment.save.top_n,
        #         smallest=True) # keep the lowest val loss, change to False if using an increasing val metric


        # Finally, log memory usage in MB
        process = psutil.Process(os.getpid())
        mem_usage = int(process.memory_info().rss / 1000000)
        data_logger.record("System/RAM Usage (MB)", mem_usage, epoch)
        print("\nEpoch {} RAM Memory Usage: {} MB\n".format(epoch, mem_usage))

        # print the gpu memory usage using nvidia smi
        print(f"\n Epoch {epoch} GPU Memory Usage: {get_gpu_usage_mb()} MB\n")

    # terminate logging
    data_logger.close()


def main(args):
    if args.config is not None:
        ext_cfg = json.load(open(args.config, 'r'))
        config = config_factory(ext_cfg["algo_name"])
        # update config with external json - this will throw errors if
        # the external config has keys not present in the base algo config
        config.unlock()
        with config.values_unlocked():
            config.unlock()
            config.update(ext_cfg)
    else:
        config = config_factory(args.algo)

    if args.dataset is not None:
        config.train.data = args.dataset
    
    config.unlock()
    if args.dataset_2 is not None:
        config.train.data_2 = args.dataset_2
    
    if args.name is not None:
        config.experiment.name = args.name
    
    if args.description is not None:
        config.experiment.description = args.description
        
    # get torch device
    device = TorchUtils.get_torch_device(try_to_use_cuda=config.train.cuda)

    # maybe modify config for debugging purposes
    if args.debug:
        # shrink length of training to test whether this run is likely to crash
        config.unlock()
        config.lock_keys()

        # train and validate (if enabled) for 3 gradient steps, for 2 epochs
        config.experiment.epoch_every_n_steps = 5
        config.train.num_epochs = 2

        config.experiment.validation_epoch_every_n_steps = 5
        config.experiment.validation_freq = 1
        config.experiment.save.video_freq = 1

        # if rollouts are enabled, try 2 rollouts at end of each epoch, with 10 environment steps
        config.experiment.rollout.rate = 1
        config.experiment.rollout.n = 2
        config.experiment.rollout.horizon = 10

        # send output to a temporary directory
        # config.train.output_dir = "/tmp/tmp_trained_models"

        config.experiment.logging.log_wandb=False
        config.experiment.logging.wandb_proj_name=None
        
    if args.no_wandb:
        config.experiment.logging.log_wandb=False
        config.experiment.logging.wandb_proj_name=None

    if args.non_goal_cond:
        config.observation.modalities.goal.rgb = []
        config.train.goal_mode = None

    if args.lr:
        config.algo.optim_params.policy.learning_rate.initial = args.lr
        
    if args.use_ddp:
        config.experiment.use_ddp = True

    # lock config to prevent further modifications and ensure missing keys raise errors
    config.lock()

    # catch error during training and print it
    res_str = "finished run successfully!"

    if args.use_ddp:
        try:
            print("Training with multiple GPUs!!")
            train_ddp(config=config)
        except:
            res_str = "run failed with error:\n{}\n\n{}".format(e, traceback.format_exc())
    else:
        try:
            train(config, device=device)
        except Exception as e:
            res_str = "run failed with error:\n{}\n\n{}".format(e, traceback.format_exc())
    print(res_str)


def train_argparse():
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
    parser.add_argument(
        "--algo",
        type=str,
        help="(optional) name of algorithm to run. Only needs to be provided if --config is not provided",
    )

    # Experiment Name (for tensorboard, saving models, etc.)
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="(optional) if provided, override the experiment name defined in the config",
    )

    # Experiment Name (for tensorboard, saving models, etc.)
    parser.add_argument(
        "--description",
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

    # 2nd Dataset path, to override the one in the config
    parser.add_argument(
        "--dataset_2",
        type=str,
        default=None,
        help="(optional) if provided, override the dataset path defined in the config",
    )


    parser.add_argument(
        "--bddl_file",
        type=str,
        default=None,
        help="(optional) if provided, the task's goal is specified as the symbolic goal in the bddl file (several symbolic predicates connected with AND / OR)",
    )

    parser.add_argument(
        "--video_prompt",
        type=str,
        default=None,
        help="(optional) if provided, a task video prompt is loaded and used in the evaluation rollouts",
    )

    # debug mode
    parser.add_argument(
        "--debug",
        action='store_true',
        help="set this flag to run a quick training run for debugging purposes"
    )
    parser.add_argument(
        "--no-wandb",
        action='store_true',
        help="set this flag to run without wandb"
    )
    parser.add_argument(
        "--use-ddp",
        action='store_true',
        help="set this flag to run on multiple gpus"
    )
    parser.add_argument(
        "--non-goal-cond",
        action='store_true',
        help="edits config to remove rgb goal conditioning"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="learning rate"
    )


    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = train_argparse()
    if "DT" not in args.description:
        time_str = f"{args.description}_DT_{datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M-%S')}"
        args.description = time_str
    main(args)

