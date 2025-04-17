#!/bin/bash
#SBATCH --job-name=run_qwen
#SBATCH --output=run_qwen_%j.out
#SBATCH --error=run_qwen_%j.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=hq48@cornell.edu  
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --get-user-env
#SBATCH --mem=64G
#SBATCH -t 40:00:00
#SBATCH --partition=ju
#SBATCH --gres=gpu:2

source /share/apps/anaconda3/2020.11/etc/profile.d/conda.sh
conda activate /home/hq48/vlm-testing/myenv

python qwen.py