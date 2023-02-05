#!/bin/bash
#SBATCH --job-name=test # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=32      # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:1      # number of gpus per node
#SBATCH --qos=debug
#SBATCH --partition=gpu
#SBATCH --time=30:00          # total run time limit (HH:MM:SS)


python main.py \
    --seed 42 \
    --devices '0' \
    --enable_logger \
    --config 'configs/config.yaml' \
    --job_type 'debug' \
    --project 'debug' \
