#!/bin/bash
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=16G
#SBATCH --cpus-per-gpu=4
#SBATCH --time=12:00:00
#SBATCH --partition=dineshj-compute
#SBATCH --qos=dj-med
#SBATCH --output=./slurm_outputs/%j.out

TASK="Isaac-CrazyflieManipulatorLong-SRT-Hover-v0"
EXPERIMENT="CrazyflieManipulator_CTATT"
RUN_NAME="sbatch_test"
NUM_STEPS=4096
AGENT_STEPS=64
SEED=0


source /home/pratikk/.bashrc
eval "$(conda shell.bash hook)"
conda activate isaaclab
python train_rslrl.py --task $TASK --num_steps $NUM_STEPS agent.num_steps_per_env=$AGENT_STEPS \
	--seed $SEED --experiment_name $EXPERIMENT --run_name $RUN_NAME \


