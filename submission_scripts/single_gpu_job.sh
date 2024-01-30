#!/bin/bash

#SBATCH --job-name 1gpu_predict


#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem=256GB
#SBATCH --partition=gpu1


#SBATCH --error=/home1/zhang2012/nowcasting/runs/job.%J.err 
#SBATCH --output=/home1/zhang2012/nowcasting/runs/job.%J.out

# module load cuda/11.8.0-gcc-9.4.0-dmftitd 

python /home1/zhang2012/nowcasting/examples/wa_imerg_convLSTM.py
# python /home1/zhang2012/nowcasting/examples/test.py