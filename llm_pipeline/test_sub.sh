#!/bin/bash
#SBATCH -N 1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=64g
#SBATCH -J "optimized inference"
#SBATCH -p short
#SBATCH -t 0-03:00:00
#SBATCH --gres=gpu:1 

module load python/3.12.10
module load cuda/12.9.0
source .venv/bin/activate
python3 test.py