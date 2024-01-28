#!/bin/bash
#SBATCH --job-name=cudaExample
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=0:01:00

module load gcc/7.3.0 cuda/10.0.130

gcc input.c -o inputGenerator
nvcc cudaExample.cu -O3 -o output

./inputGenerator 10000
./output 10000
