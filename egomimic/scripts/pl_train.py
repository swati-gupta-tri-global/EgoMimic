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

python3 scripts/pl_train.py --config configs/egomimic_combined_lbm_egodex.json --dataset ../datasets/LBM_sim_egocentric/processed/BimanualHangMugsOnMugHolderFromDryingRack.hdf5 --dataset_2 ../datasets/egodex/processed/test/fold_stack_unstack_unfold_cloths.hdf5 --debug 
datasets/LBM_sim_egocentric/processed/BimanualPlacePearFromBowlOnCuttingBoard.hdf5


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 scripts/pl_train.py   --config configs/egomimic_combined_lbm_egodex.json   --dataset ../datasets/LBM_sim_egocentric/small_split   --dataset_2 ../datasets/egodex/small_split   --debug --gpus-per-node 8 --num-nodes 1

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python3 scripts/pl_train.py \
  --config configs/egomimic_combined_lbm_egodx.json \
  --dataset ../datasets/LBM_sim_egocentric/small_split \
  --dataset_2 ../datasets/egodx/small_split \
  --debug \
  --gpus-per-node 8 \
  --num-nodes 1

# You requested gpu: [0, 1, 2, 3, 4, 5, 6, 7]
 But your machine only has: [0, 1, 2, 3]

NCCL_SOCKET_IFNAME=bond0 NCCL_DEBUG=INFO NCCL_P2P_DISABLE=1 TORCH_NCCL_BLOCKING_WAIT=1 NCCL_IB_DISABLE=1 python3 scripts/pl_train.py  --config configs/egomimic_combined_lbm_egodex.json --gpus-per-node 4   --num-nodes 1

NCCL_DEBUG=INFO python3 scripts/pl_train.py  --config configs/egomimic_combined_lbm_egodex.json --gpus-per-node 4  --num-nodes 1

torchrun --nproc_per_node=4 scripts/pl_train.py --config configs/egomimic_combined_lbm_egodex.json --gpus-per-node 4 --num-nodes 1

# trained 40 epochs with sample data: /workspace/externals/EgoMimic/trained_models_highlevel/test/None_DT_2025-11-13-02-40-03/models/last.ckpt
NCCL_DEBUG=INFO python3 scripts/pl_train.py  --config configs/egomimic_combined_lbm_egodex.json --gpus-per-node 2   --num-nodes 1 --ckpt /workspace/externals/EgoMimic/trained_models_highlevel/test/None_DT_2025-11-13-02-40-03/models/last.ckpt --eval

# trained full run
torchrun --nproc_per_node=4 scripts/pl_train.py  --config configs/egomimic_combined_lbm_egodex.json --gpus-per-node 4   --num-nodes 1 --ckpt /workspace/externals/EgoMimic/trained_models_highlevel/test/None_DT_2025-11-22-18-26-12/models/model_epoch_epoch=9.ckpt

# eval
torchrun --nproc_per_node=4 scripts/pl_train.py  --config configs/egomimic_combined_lbm_egodex.json --gpus-per-node 4   --num-nodes 1 --ckpt /workspace/externals/EgoMimic/trained_models_highlevel/test/None_DT_2025-11-22-18-26-12/models/model_epoch_epoch=9.ckpt --eval
CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 --master-port=29501 scripts/pl_train.py  --config configs/egomimic_combined_lbm_egodex.json --gpus-per-node 2 --num-workers 2  --no-eval-videos --num-nodes 1 --ckpt /workspace/externals/EgoMimic/trained_models_highlevel/test/None_DT_2025-11-22-18-26-12/models/model_epoch_epoch=39.ckpt --eval
"""
import argparse
import json
import os
import traceback
from egomimic.configs import config_factory
import datetime
import time
from egomimic.utils.egomimicUtils import nds
import matplotlib.pyplot as plt
from egomimic.pl_utils.pl_train_utils import train, eval
from egomimic.pl_utils.pl_data_utils import json_to_config
import torch
import numpy as np
import egomimic


def main(args):
    # set random seeds
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    # numpy seeds
    np.random.seed(0)


    if args.config is not None:
        ext_cfg = json.load(open(args.config, "r"))
        config = config_factory(ext_cfg["algo_name"])
        # update config with external json - this will throw errors if
        # the external config has keys not present in the base algo config
        with config.values_unlocked():
            config.update(ext_cfg)
    elif args.ckpt_path is not None:
        ckpt = torch.load(args.ckpt_path, map_location="cpu")
        config = json_to_config(ckpt["hyper_parameters"]["config_json"])
    else:
        assert False, "Must provide a config file or a ckpt path"

    # if args.resume_dir is not None:
    if args.name is not None and args.description is not None:
        base_output_dir = os.path.dirname(egomimic.__file__)
        possible_resume = os.path.join(base_output_dir, "../trained_models_highlevel", args.name, args.description, "models/last.ckpt")
        # possible_resume = os.path.join(args.resume_dir, "models/last.ckpt")
        print(f"TRYING TO RESUME FROM: {possible_resume}")
        if os.path.exists(possible_resume) or os.path.islink(possible_resume):
            args.ckpt_path = possible_resume
            print("RESUMING FROM LAST CHECKPOINT")    

    if args.dataset is not None:
        config.train.data = args.dataset

    if args.dataset_2 is not None:
        config.train.data_2 = args.dataset_2

    if args.alternate_val:
        config.train.alternate_val = args.alternate_val

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
    
    if args.hand_lambda is not None:
        config.algo.sp.hand_lambda = args.hand_lambda

    if args.batch_size:
        config.train.batch_size = args.batch_size
    
    if args.num_workers is not None:
        config.train.num_data_workers = args.num_workers
    
    if args.no_eval_videos:
        config.experiment.save_eval_videos = False
    
    if args.train_key:
        config.train.hdf5_filter_key = args.train_key

    if args.train_key_2:
        config.train.hdf5_2_filter_key = args.train_key_2

    if args.brightness is not None:
        config.observation.encoder.rgb.obs_randomizer_kwargs.brightness_min = (
            args.brightness[0]
        )
        config.observation.encoder.rgb.obs_randomizer_kwargs.brightness_max = (
            args.brightness[1]
        )

    if args.contrast is not None:
        config.observation.encoder.rgb.obs_randomizer_kwargs.contrast_min = (
            args.contrast[0]
        )
        config.observation.encoder.rgb.obs_randomizer_kwargs.contrast_max = (
            args.contrast[1]
        )

    if args.saturation is not None:
        config.observation.encoder.rgb.obs_randomizer_kwargs.saturation_min = (
            args.saturation[0]
        )
        config.observation.encoder.rgb.obs_randomizer_kwargs.saturation_max = (
            args.saturation[1]
        )

    if args.hue is not None:
        config.observation.encoder.rgb.obs_randomizer_kwargs.hue_min = args.hue[0]
        config.observation.encoder.rgb.obs_randomizer_kwargs.hue_max = args.hue[1]

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
        config.experiment.logging.log_wandb = False
        config.experiment.logging.wandb_proj_name = None

        config.experiment.validation_max_samples = 200
        config.experiment.validation_freq = 2
        config.experiment.save.every_n_epochs = 2
        config.experiment.save.video_freq = 2
        config.experiment.name = "debug_run"
        config.train.gpus_per_node = 1
        config.train.num_nodes = 1
        config.train.num_data_workers = 0
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
        config.experiment.logging.log_wandb = False
        config.experiment.logging.wandb_proj_name = None
    else:
        config.wandb_project_name = args.wandb_project_name
        config.train.fast_dev_run = False

    if config.train.gpus_per_node == 1 and args.num_nodes == 1:
        os.environ["OMP_NUM_THREADS"] = "1"

    if args.no_wandb:
        config.experiment.logging.log_wandb = False
        config.experiment.logging.wandb_proj_name = None

    assert (
        config.experiment.validation_freq % config.experiment.save.every_n_epochs == 0
    ), "current code expects validation_freq to be a multiple of save.every_n_epochs"
    assert (
        config.experiment.validation_freq == config.experiment.save.video_freq
    ), "current code expects validation_freq to be the same as save.video_freq"

    # lock config to prevent further modifications and ensure missing keys raise errors
    config.lock()

    # catch error during training and print it
    res_str = "finished run successfully!"
    important_stats = None
    try:
        if args.eval:
            # Handle dataset path for evaluation
            if args.dataset is not None:
                config.unlock()
                config.train.data = args.dataset
                config.lock()
            
            eval(config, args.ckpt_path, type=config.train.data_type)
            return
        else:
            important_stats = train(config, args.ckpt_path)
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

    # Dataset paths, to override the one in the config
    parser.add_argument(
        "--dataset",
        type=str,
        nargs='+',
        default=None,
        help="(optional) if provided, override the dataset path(s) defined in the config. "
             "Can specify multiple HDF5 files, directories containing HDF5 files, or glob patterns.",
    )

    # Dataset paths, to override the one in the config
    parser.add_argument(
        "--dataset_2",
        type=str,
        nargs='+',
        default=None,
        help="(optional) if provided, override the dataset path(s) defined in the config. "
             "Can specify multiple HDF5 files, directories containing HDF5 files, or glob patterns.",
    )

    parser.add_argument(
        "--alternate-val", type=str, default=None, help="alternate validation dataset"
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

    parser.add_argument("--lr", type=float, default=None, help="learning rate")

    parser.add_argument("--hand-lambda", type=float, default=None, help="hand data weighting")

    parser.add_argument("--batch-size", type=int, default=None, help="batch size")

    parser.add_argument("--num-workers", type=int, default=None, help="number of DataLoader workers (default: 16, use 0-4 if OOM issues)")

    parser.add_argument(
        "--no-eval-videos", action="store_true", help="disable saving evaluation videos to speed up evaluation"
    )

    parser.add_argument(
        "--wandb_project_name",
        type=str,
        default="egoplay",
    )

    parser.add_argument(
        "--ckpt-path",
        type=str,
        default=None,
        help="path to pytorch lightning ckpt file",
    )

    parser.add_argument(
        "--eval", action="store_true", help="set this flag to run a evaluation"
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
        "--no-wandb", action="store_true", help="set this flag to run a without wandb"
    )

    parser.add_argument("--overcap", action="store_true", help="overcap partition")

    parser.add_argument('--partition', type=str, default='hoffman-lab', help='partition')

    parser.add_argument(
        "--brightness", nargs=2, help="brightness min and max", default=None, type=float
    )

    parser.add_argument(
        "--contrast", nargs=2, help="contrast min and max", default=None, type=float
    )

    parser.add_argument(
        "--saturation", nargs=2, help="saturation min and max", default=None, type=float
    )

    parser.add_argument(
        "--hue", nargs=2, help="hue min and max", default=None, type=float
    )

    parser.add_argument(
        "--train-key", type=str, default=None
    )

    parser.add_argument(
        "--train-key-2", type=str, default=None
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = train_argparse()

    if not args.eval and (args.description is None or "DT" not in args.description):
        time_str = f"{args.description}_DT_{datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M-%S')}"
        args.description = time_str

    main(args)
