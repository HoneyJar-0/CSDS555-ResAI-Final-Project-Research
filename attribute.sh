#!/bin/bash
#SBATCH -N 1                      # (--nodes) allocate compute nodes
#SBATCH --cpus-per-task=4         # (--ntasks) require 4 CPUs per task (process)
#SBATCH --mem=100g                # allocate memory
#SBATCH -J "ResAIRun%j"	            # (--job-name)
#SBATCH -o console%j.out          # (--output) name the output file
#SBATCH -e console%j.err          # (--error) name the error file
#SBATCH -p academic               # (--partition) partition to submit
#SBATCH --gres=gpu:1              # request GPUs
#SBATCH --time=0-05:00:00	        # Set time limit of 2 days to avoid short durations

mkdir verification_logs/
module load python/3.12.10
module load cuda/12.9.0 
python3 -m venv ResAIProject
source ResAIProject/bin/activate
pip3 install -r requirements.txt 
python3 attribute_uuid_verification.py &
wait
mv *.out verification_logs/
mv *.err verification_logs/
