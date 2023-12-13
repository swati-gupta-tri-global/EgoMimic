import argparse
import submitit
from mimicplay.scripts.train import main


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
        required=True,
        help="override the experiment name defined in the config",
    )

    # Dataset path, to override the one in the config
    parser.add_argument(
        "--dataset",
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

    args = parser.parse_args()


    # the AutoExecutor class is your interface for submitting function to a cluster or run them locally.
    # The specified folder is used to dump job information, logs and result when finished
    # %j is replaced by the job id at runtime
    assert args.name is not None, "Must provide a name for the experiment"
    log_folder = f"/coc/flash9/skareer6/Projects/EgoPlay/EgoPlay/mimicplay/scripts/exps/{args.name}/%j"
    executor = submitit.AutoExecutor(folder=log_folder)
    # The AutoExecutor provides a simple abstraction over SLURM to simplify switching between local and slurm jobs (or other clusters if plugins are available).
    # specify sbatch parameters (here it will timeout after 4min, and run on dev)
    # This is where you would specify `gpus_per_node=1` for instance
    # Cluster specific options must be appended by the cluster name:
    # Eg.: slurm partition can be specified using `slurm_partition` argument. It
    # will be ignored on other clusters:
    executor.update_parameters(slurm_partition="hoffman-lab", cpus_per_task=13, nodes=1, slurm_ntasks_per_node=1, gpus_per_node="a40:1", slurm_qos="short", slurm_mem_per_gpu="40G", timeout_min=60*24*2)
    # The submission interface is identical to concurrent.futures.Executor

    job = executor.submit(main, args)