#!/bin/bash
#SBATCH -N 1                      # (--nodes) allocate compute nodes
#SBATCH --cpus-per-task=4         # (--ntasks) require 4 CPUs per task (process)
#SBATCH -n 8			                # number of CPUS
#SBATCH --mem=100g                # allocate memory
#SBATCH -J "ResAIRun"	            # (--job-name)
#SBATCH -o console%j.out          # (--output) name the output file
#SBATCH -e console%j.err          # (--error) name the error file
#SBATCH -p academic               # (--partition) partition to submit
#SBATCH --gres=gpu:1              # request GPUs
#SBATCH --time=2-00:00:00	        # Set time limit of 2 days to avoid short durations
requeue_handler() {
  echo "Job received SIGTERM, requeueing"
  mv *.out logs/
  mv *.err logs/
  mv notice.txt
  sbatch submission.sh
  exit 0 # Exit cleanly after cleanup
}

trap 'requeue_handler' SIGTERM


module load python/3.11.10
module load cuda/12.4.0/3mdaov5        	  # load Python 3.11.10 (highest compatible with PyTorch)
python3 -m venv ResAIProject	          # create virtual environment
source ResAIProject/bin/activate   	  # activate virtual environment
pip install -r requirements.txt           # install from requirements.txt. Alternatively, list out the pip3 installs
python3 main.py &
wait
