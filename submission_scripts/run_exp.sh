#!/bin/bash

#SBATCH --job-name nosep_01


#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem=256GB
#SBATCH --partition=gpu1

#SBATCH --error=/home1/zhang2012/nowcasting/runs/job.%J.err 
#SBATCH --output=/home1/zhang2012/nowcasting/runs/job.%J.out

export CUDA_VISIBLE_DEVICES=1
python /home1/zhang2012/nowcasting/examples/predict_2datasets.py
