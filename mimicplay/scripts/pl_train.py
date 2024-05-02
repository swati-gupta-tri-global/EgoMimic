# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].
"""
The main entry point for training policies. Adapted to use PyTorch Lightning and Optimus codebase.

Args:
    config (str): path to a config json that will be used to override the default settings.
        If omitted, default settings are used. This is the preferred way to run experiments.

    algo (str): name of the algorithm to run. Only needs to be provided if @config is not
        provided.

    name (str): if provided, override the experiment name defined in the config

    dataset (str): if provided, override the dataset path defined in the config

    debug (bool): set this flag to run a quick training run for debugging purposes
"""
import argparse
import json
import os
import sys
import traceback
from collections import OrderedDict

import numpy as np
import psutil
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.tensor_utils as TensorUtils
import torch
from torch.utils.data import DataLoader
import wandb
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import Callback, StochasticWeightAveraging, Timer, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.profilers import AdvancedProfiler
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities import rank_zero_only


import robomimic.utils.train_utils as TrainUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
from robomimic.utils.log_utils import PrintLogger, DataLogger
from robomimic.algo.algo import PolicyAlgo

from mimicplay.configs import config_factory
from mimicplay.algo import algo_factory, RolloutPolicy
from mimicplay.utils.train_utils import get_exp_dir, rollout_with_stats, load_data_for_training

import mimicplay.utils.val_utils as ValUtils

import datetime
import time
from mimicplay.scripts.aloha_process.simarUtils import nds

class DataModuleWrapper(LightningDataModule):
    """
    Wrapper around a LightningDataModule that allows for the data loader to be refreshed
    constantly.
    """

    def __init__(
        self,
        train_dataset,
        valid_dataset,
        train_dataloader_params,
        valid_dataloader_params,
    ):
        """
        Args:
            data_module_fn (function): function that returns a LightningDataModule
        """
        super().__init__()
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.train_dataloader_params = train_dataloader_params
        self.valid_dataloader_params = valid_dataloader_params

    def train_dataloader(self):
        new_dataloader = DataLoader(dataset=self.train_dataset, **self.train_dataloader_params)
        return new_dataloader
    
    def val_dataloader(self):
        new_dataloader = DataLoader(dataset=self.valid_dataset, **self.valid_dataloader_params)
        return new_dataloader


class ModelWrapper(LightningModule):
    """
    Wrapper class around robomimic models to ensure compatibility with Pytorch Lightning.
    """

    def __init__(self, model, datamodule):
        """
        Args:
            model (PolicyAlgo): robomimic model to wrap.
        """
        super().__init__()
        self.model = model
        self.nets = (
            self.model.nets
        )  # to ensure the lightning module has access to the model's parameters
        try:
            self.params = self.model.nets["policy"].params
        except:
            pass
        self.step_log_all_train = []
        self.step_log_all_valid = []

        self.datamodule = datamodule

    def training_step(self, batch, batch_idx):
        self.train()
        batch["obs"] = ObsUtils.process_obs_dict(batch["obs"])
        info = PolicyAlgo.train_on_batch(self.model, batch, self.current_epoch, validate=False)
        batch = self.model.process_batch_for_training(batch)
        predictions = self.model._forward_training(batch)
        losses = self.model._compute_losses(predictions, batch)
        info["losses"] = TensorUtils.detach(losses)
        self.step_log_all_train.append(self.model.log_info(info))

        # count=0
        # for name, param in self.named_parameters():
        #     if param.grad is None:
        #         count += 1
        #         # print(name)
        # print("Unused params: ", count)

        if self.global_step % self.model.global_config.experiment.epoch_every_n_steps == 0:
            # flatten and take the mean of the metrics
            log = {}
            for i in range(len(self.step_log_all_train)):
                for k in self.step_log_all_train[i]:
                    if k not in log:
                        log[k] = []
                    log[k].append(self.step_log_all_train[i][k])
            log_all = dict((k, float(np.mean(v))) for k, v in log.items())
            for k in self.model.optimizers:
                for i, param_group in enumerate(self.model.optimizers[k].param_groups):
                    log_all["Optimizer/{}{}_lr".format(k, i)] = param_group["lr"]
            for k, v in log_all.items():
                self.log("Train/" + k, v, sync_dist=True)
            self.step_log_all_train = []
        return losses["action_loss"]

    def val_dataloader(self):
        return self.datamodule.val_dataloader()

    def configure_optimizers(self):
        if self.model.lr_schedulers["policy"]:
            lr_scheduler_dict = {
                "scheduler": self.model.lr_schedulers["policy"],
                "interval": "step",
                "frequency": 1,
            }
            return {
                "optimizer": self.model.optimizers["policy"],
                "lr_scheduler": lr_scheduler_dict,
            }
        else:
            return {
                "optimizer": self.model.optimizers["policy"],
            }

    # def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
    #     optimizer.zero_grad(optimizer_idx)

    def on_train_epoch_start(self):
        valid_step_log = {"final_mse_avg": 0.0}
        if self.global_rank == 0:
            # Perform custom validation
            val_freq = self.model.global_config.experiment.validation_freq
            video_freq = self.model.global_config.experiment.save.video_freq


            with torch.no_grad():
                if self.current_epoch % val_freq == 0 and self.current_epoch != 0:
                    self.eval()
                    self.zero_grad()
                    pass_vid = os.path.join(self.trainer.default_root_dir, "videos") if self.current_epoch % video_freq == 0 else None
                    valid_step_log = ValUtils.evaluate_high_level_policy(self.model, self.val_dataloader(), pass_vid, max_samples=self.model.global_config.experiment.validation_max_samples) #save vid only once every video_freq epochs
                    self.train()

        self.log("final_mse_avg", valid_step_log["final_mse_avg"], sync_dist=True, reduce_fx="max")


        # Finally, log memory usage in MB
        process = psutil.Process(os.getpid())
        mem_usage = process.memory_info().rss / int(1e9)
        print("\nEpoch {} Memory Usage: {} GB\n".format(self.current_epoch, mem_usage))
        # self.log('epoch', self.trainer.current_epoch)
        self.log("System/RAM Usage (GB)", mem_usage, sync_dist=True)

        return super().on_train_epoch_start()

    def lr_scheduler_step(self, scheduler, optimizer_idx):
        if False and self.model.lr_warmup:
            # lr warmup schedule taken from Gato paper
            # update params
            initial_lr = 0
            target_lr = self.model.optim_params["policy"]["learning_rate"]["initial"]
            # manually warm up lr without a scheduler
            schedule_iterations = 10000
            if self.global_step < schedule_iterations:
                for pg in self.optimizers().param_groups:
                    pg["lr"] = (
                        initial_lr
                        + (target_lr - initial_lr) * self.global_step / schedule_iterations
                    )
            else:
                scheduler.step(self.global_step - schedule_iterations)
        else:
            scheduler.step(self.global_step)


def train(config, ckpt_path, resume_dir):
    RANK = os.environ["SLURM_PROCID"]
    torch.set_float32_matmul_precision("medium")
    seed_everything(config.train.seed, workers=True)
    """
    Train a model using the algorithm.
    """

    if ckpt_path is not None:
        ext_cfg = json.load(open(os.path.join(resume_dir, "config.json"), "r"))
        config = config_factory(ext_cfg["algo_name"])
        # update config with external json - this will throw errors if
        # the external config has keys not present in the base algo config
        with config.values_unlocked():
            config.update(ext_cfg)
        log_dir, ckpt_dir, video_dir, rollout_dir, exp_log_dir = (
            os.path.join(resume_dir, "logs"),
            os.path.join(resume_dir, "models"),
            os.path.join(resume_dir, "videos"),
            os.path.join(resume_dir, "rollouts"),
            os.path.join(resume_dir, "exp_logs"),
        )
        config.lock()
    else:
        print("\n============= New Training Run with Config =============")
        print(config)
        print("")
        log_dir, ckpt_dir, video_dir, time_str = get_exp_dir(config, rank=int(RANK))
        base_output_dir = os.path.join(config.train.output_dir, config.experiment.name)
        exp_dir = os.path.join(base_output_dir, time_str)
        rollout_dir = os.path.join(base_output_dir, time_str, "rollouts")
        exp_log_dir = os.path.join(base_output_dir, time_str, "exp_logs")
        if RANK == 0:
            os.makedirs(rollout_dir, exist_ok=True)
            os.makedirs(exp_log_dir, exist_ok=True)

    # if config.experiment.logging.terminal_output_to_txt:
    #     # log stdout and stderr to a text file
    #     logger = PrintLogger(os.path.join(log_dir, "log.txt"))
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
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=config.train.data)
    shape_meta = FileUtils.get_shape_metadata_from_dataset(
        dataset_path=dataset_path,
        all_obs_keys=config.all_obs_keys,
        verbose=True,
        ac_key=config.train.ac_key
    )

    if type(env_meta) is list:
        env_metas = env_meta
    else:
        env_metas = [env_meta]
    # create environment

    # setup for a new training runs
    model = algo_factory(
        algo_name=config.algo_name,
        config=config,
        obs_key_shapes=shape_meta["all_shapes"],
        ac_dim=shape_meta["ac_dim"],
        device="cuda"  # default to cpu, pl will move to gpu
    )

    # if config.train.ckpt_path is not None:
    #     model = ModelWrapper.load_from_checkpoint(config.train.ckpt_path, model=model).model

    # save the config as a json file
    if RANK == 0:
        with open(os.path.join(log_dir, "..", "config.json"), "w") as outfile:
            json.dump(config, outfile, indent=4)

    print("\n============= Model Summary =============")
    print(model)  # print model summary
    print("")

    # load training data
    trainset, validset = load_data_for_training(
        config, obs_keys=shape_meta["all_obs_keys"]
    )
    print("\n============= Training Dataset =============")
    print(trainset)
    print("")

    # maybe retreve statistics for normalizing observations
    obs_normalization_stats = None
    # if config.train.hdf5_normalize_obs:
    #     obs_normalization_stats = trainset.get_obs_normalization_stats()

    loggers = [] if config.experiment.logging.wandb_proj_name is None else [WandbLogger(
        project=config.experiment.logging.wandb_proj_name,
        sync_tensorboard=True,
        name=config.experiment.description,
        config=config,
        save_dir=log_dir,
    )]

    # breakpoint()
    callbacks = [
        ModelCheckpoint(
            every_n_epochs=config.experiment.save.every_n_epochs,
            dirpath=ckpt_dir,
            save_on_train_epoch_end=True,
            filename="model_epoch_{epoch}",
            save_top_k=-1,
        ),
        ModelCheckpoint(
            dirpath=ckpt_dir,
            save_on_train_epoch_end=True,
            filename="model_epoch_{epoch}_{final_mse_avg:.1f}",
            save_top_k=3,
            monitor="final_mse_avg",
            mode="min",
        )
    ]
    # if config.train.use_swa:
    #     callbacks.append(
    #         StochasticWeightAveraging(swa_lrs=config.algo.optim_params.policy.learning_rate.initial)
    #     )
    trainer = Trainer(
        max_epochs=config.train.num_epochs,
        limit_train_batches=config.experiment.epoch_every_n_steps,
        accelerator="gpu",
        devices=config.train.gpus_per_node,
        num_nodes=config.train.num_nodes,
        logger=loggers,
        default_root_dir=exp_dir,
        callbacks=callbacks,
        fast_dev_run=config.train.fast_dev_run,
        # val_check_interval=config.experiment.validation_epoch_every_n_steps,
        check_val_every_n_epoch=config.experiment.validation_freq,
        # gradient_clip_algorithm="norm",
        # gradient_clip_val=config.train.max_grad_norm,
        # precision=16 if config.train.amp_enabled else 32,
        precision=32,
        reload_dataloaders_every_n_epochs=0,
        use_distributed_sampler=True,
        # strategy=DDPStrategy(
        #     find_unused_parameters=False,
        #     static_graph=True,
        #     gradient_as_bucket_view=True,
        # ),
        strategy="ddp_find_unused_parameters_true",
        profiler="simple",
        # profiler=AdvancedProfiler(dirpath=".", filename="perf_logs")
        # if args.profiler != "none"
        # else None,
    )

    train_sampler = trainset.get_dataset_sampler()
    valid_sampler = validset.get_dataset_sampler()

    datamodule=DataModuleWrapper(
        train_dataset=trainset,
        valid_dataset=validset,
        train_dataloader_params=dict(
            sampler=train_sampler,
            batch_size=config.train.batch_size,
            shuffle=(train_sampler is None),
            num_workers=config.train.num_data_workers,
            drop_last=True,
            pin_memory=True,
        ),
        valid_dataloader_params=dict(
            sampler=valid_sampler,
            batch_size=config.train.batch_size,
            shuffle=False,
            num_workers=config.train.num_data_workers,
            drop_last=True,
            pin_memory=True,
        ),
    )
    model=ModelWrapper(model, datamodule)


    trainer.fit(
        model=model,
        datamodule=datamodule,
        ckpt_path=ckpt_path,
    )


def main(args):
    if args.config is not None:
        ext_cfg = json.load(open(args.config, "r"))
        config = config_factory(ext_cfg["algo_name"])
        # update config with external json - this will throw errors if
        # the external config has keys not present in the base algo config
        with config.values_unlocked():
            config.update(ext_cfg)
    else:
        config = config_factory(args.algo)

    if args.dataset is not None:
        config.train.data = args.dataset

    if args.output_dir is not None:
        config.train.output_dir = args.output_dir

    if args.name is not None:
        config.experiment.name = args.name

    if args.seed is not None:
        config.train.seed = args.seed
    
    if args.description is not None:
        config.experiment.description = args.description
    
    if args.lr:
        config.algo.optim_params.policy.learning_rate.initial = args.lr

    if args.batch_size:
        config.train.batch_size = args.batch_size

    config.train.gpus_per_node = args.gpus_per_node
    config.train.num_nodes = args.num_nodes
    # maybe modify config for debugging purposes
    if args.debug:
        # shrink length of training to test whether this run is likely to crash
        config.unlock()
        config.lock_keys()

        # train and validate (if enabled) for 1 gradient steps, for 2 epochs
        # config.train.fast_dev_run = 2
        config.train.num_epochs = 10
        config.experiment.save.every_n_epochs = 5


        # if rollouts are enabled, try 10 rollouts at end of each epoch, with 10 environment steps
        config.experiment.epoch_every_n_steps = 10

        # send output to a temporary directory
        config.experiment.logging.log_wandb=False
        config.experiment.logging.wandb_proj_name=None

        config.experiment.validation_max_samples = 1000
        config.experiment.validation_freq = 2
        config.experiment.save.every_n_epochs = 2
        config.experiment.save.video_freq = 2
    elif args.profiler != "none":
        # shrink length of training to test whether this run is likely to crash
        config.unlock()
        config.lock_keys()

        config.experiment.epoch_every_n_steps = 10
        config.train.num_epochs = 1
        config.train.num_data_workers = 0

        # if rollouts are enabled, try 2 rollouts at end of each epoch, with 10 environment steps
        # config.experiment.rollout.rate = 1
        # config.experiment.rollout.n = 1

        # send output to a temporary directory
        config.experiment.logging.log_wandb=False
        config.experiment.logging.wandb_proj_name=None
    else:
        config.wandb_project_name = args.wandb_project_name
        config.train.fast_dev_run = False

    if config.train.gpus_per_node == 1 and args.num_nodes == 1:
        os.environ["OMP_NUM_THREADS"] = "1"
    
    if args.no_wandb:
        config.experiment.logging.log_wandb=False
        config.experiment.logging.wandb_proj_name=None
    
    assert config.experiment.validation_freq % config.experiment.save.every_n_epochs == 0, "current code expects validation_freq to be a multiple of save.every_n_epochs"
    assert config.experiment.validation_freq == config.experiment.save.video_freq, "current code expects validation_freq to be the same as save.video_freq"

    # lock config to prevent further modifications and ensure missing keys raise errors
    config.lock()

    # catch error during training and print it
    res_str = "finished run successfully!"
    important_stats = None
    try:
        important_stats = train(config, args.ckpt_path, args.resume_dir)
        important_stats = json.dumps(important_stats, indent=4)
    except Exception as e:
        res_str = "run failed with error:\n{}\n\n{}".format(e, traceback.format_exc())
    print(res_str)
    if important_stats is not None:
        print("\nRollout Success Rate Stats")
        print(important_stats)

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

    # description
    parser.add_argument(
        "--description",
        type=str,
        default=None,
        help="description",
    )

    # Dataset path, to override the one in the config
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="(optional) if provided, override the dataset path defined in the config",
    )

    # Output path, to override the one in the config
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="(optional) if provided, override the output path defined in the config",
    )

    # debug mode
    parser.add_argument(
        "--debug",
        action="store_true",
        help="set this flag to run a quick training run for debugging purposes",
    )

    # env seed
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="(optional) if provided, sets the seed",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="learning rate"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="batch size"
    )

    parser.add_argument(
        "--wandb_project_name",
        type=str,
        default="egoplay",
    )

    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="path to pytorch lightning ckpt file",
    )

    parser.add_argument(
        "--resume_dir",
        type=str,
        default=None,
        help="path to pytorch lightning resume dir",
    )

    parser.add_argument(
        "--profiler",
        type=str,
        default="none",
        help="profiler to use (none, pytorch, simple, advanced)",
    )

    parser.add_argument(
        "--gpus-per-node",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--num-nodes",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--no-wandb",
        action='store_true',
        help="set this flag to run a without wandb"
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = train_argparse()

    if "DT" not in args.description:
        time_str = f"{args.description}_DT_{datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M-%S')}"
        args.description = time_str

    main(args)