#!/bin/bash
#SBATCH -N 1      
#SBATCH --cpus-per-gpu=8 
#SBATCH --mem=100g
#SBATCH -J "generating"	
#SBATCH -o console%j.out
#SBATCH -e console%j.err 
#SBATCH -p academic  
#SBATCH --gres=gpu:2 
#SBATCH --time=2-00:00:00	
#SBATCH --requeue
#SBATCH --open-mode=append

## REMOVE notice.txt; no longer needed

trap 'scontrol requeue $SLURM_JOB_ID; exit 0' SIGTERM

module load python/3.12.10
module load cuda/12.9.0 
source ResAIProject/bin/activate
python3 main.py &
wait
mv *.out console_logs/
mv *.err console_logs/
