import argparse
import submitit
from mimicplay.scripts.train import main, train_argparse
import os
import datetime
import time

if __name__ == "__main__":
    os.environ["TQDM_DISABLE"]="1"

    args = train_argparse()


    # the AutoExecutor class is your interface for submitting function to a cluster or run them locally.
    # The specified folder is used to dump job information, logs and result when finished
    # %j is replaced by the job id at runtime
    assert args.name is not None, "Must provide a name for the experiment"
    assert args.description is not None, "Must provide a description for the experiment"
    base_dir = f"/coc/flash9/skareer6/Projects/EgoPlay/EgoPlay/trained_models_highlevel/{args.name}/"
    if "DT" not in args.description:
        time_str = f"{args.description}_DT_{datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M-%S')}"
        args.description = time_str
    log_dir = os.path.join(base_dir, args.description, "slurm")
    os.makedirs(log_dir)

    executor = submitit.AutoExecutor(folder=log_dir)
    # The AutoExecutor provides a simple abstraction over SLURM to simplify switching between local and slurm jobs (or other clusters if plugins are available).
    # specify sbatch parameters (here it will timeout after 4min, and run on dev)
    # This is where you would specify `gpus_per_node=1` for instance
    # Cluster specific options must be appended by the cluster name:
    # Eg.: slurm partition can be specified using `slurm_partition` argument. It
    # will be ignored on other clusters:

    executor.update_parameters(slurm_partition="hoffman-lab", cpus_per_task=13, nodes=1, slurm_ntasks_per_node=1, gpus_per_node="a40:1", slurm_qos="short", slurm_mem_per_gpu="40G", timeout_min=60*24*2)
    # The submission interface is identical to concurrent.futures.Executor

    job = executor.submit(main, args)