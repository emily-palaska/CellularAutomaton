#!/bin/bash
#SBATCH --job-name=cudaExample
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=0:01:00

module load gcc/7.3.0 cuda/10.0.130

nvcc cudaExample.cu -O3 -o output
./output
