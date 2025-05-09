#!/bin/bash
#SBATCH --job-name=run_llava
#SBATCH --output=run_llava_%j.out
#SBATCH --error=run_llava_%j.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=hq48@cornell.edu  
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --get-user-env
#SBATCH --mem=128G
#SBATCH -t 40:00:00
#SBATCH --partition=ju
#SBATCH --gres=gpu:2

source /share/apps/anaconda3/2020.11/etc/profile.d/conda.sh
conda activate /home/hq48/vlm-testing/llava

python3.11 llava_predictions.py