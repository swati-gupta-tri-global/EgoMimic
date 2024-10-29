#!/bin/bash -l

# SLURM SUBMIT SCRIPT
#SBATCH --nodes=2             # This needs to match Trainer(num_nodes=...)
#SBATCH --gres=gpu:a40:4
#SBATCH --ntasks-per-node=4   # This needs to match Trainer(devices=...)
#SBATCH --mem=0
#SBATCH --time=0-02:00:00
#SBATCH --output=pl_submit.out
#SBATCH --error=pl_submit.err

# activate conda env
conda activate eplay2

# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

# on your cluster you might need these:
# set the network interface
# export NCCL_SOCKET_IFNAME=^docker0,lo

# might need the latest CUDA
# module load NCCL/2.4.7-1-cuda.10.0

# run script from above
cd /coc/flash9/skareer6/Projects/EgoPlay/EgoPlay/mimicplay
srun python scripts/pl_train.py --config configs/act.json --dataset /coc/flash7/datasets/egoplay/oboov2_robot_apr16/oboov2_robot_apr16ACT.hdf5 --name pldebug --description debug --num_gpus 4