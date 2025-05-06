#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --reservation=fri
#SBATCH --job-name=gray-scott
#SBATCH --gpus=1
#SBATCH --output=out_gray.log

module load CUDA
nvcc -O2 -lm sample.cu -o sample
srun sample 256
srun sample 512
srun sample 1024
srun sample 2048
srun sample 4096