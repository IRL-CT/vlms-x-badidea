# Cluster Usage Guide for Running Machine Learning Models

This document serves as a personal reference for using G2 cluster, including **submitting jobs via Slurm, setting up Conda environments, using GPUs, and troubleshooting common issues**. For offical lab references, please see: https://github.com/FAR-Lab/Almanac/blob/main/wiki/tech-howto/G2-Access.md

## 1. Connecting to the Cluster

To access the cluster, use SSH:

```bash
ssh hq48@g2-login-01
```

Replace `hq48` with your netid, then enter email password. This logs you into the login node where you can prepare jobs but cannot run them directly. All computations must be submitted to a Slurm job queue.

## 2. Setup Conda environment
Create a conda env in your folder everytime you start your work:
### Create and Activate a Conda Environment
```bash
source /share/apps/anaconda3/2020.11/etc/profile.d/conda.sh
conda create --prefix /home/hq48/[foloder-name]/[file-name]
conda activate /home/hq48/vlm-testing/myenv

# llava env
conda create --prefix /home/hq48/vlm-testing/llava python=3.9 -y
# Activate the llava environment
conda activate /home/hq48/vlm-testing/llava
```
### If conda is Not Found

First, initialize Conda:
```bash
/share/apps/anaconda3/2020.11/bin/conda init
source ~/.bashrc
```
### If conda is still not found, add it to your PATH manually:
```bash
export PATH="/share/apps/anaconda3/2020.11/bin:$PATH"
```
Then, try activating the environment again:
```bash
source /share/apps/anaconda3/2020.11/etc/profile.d/conda.sh
conda activate /home/hq48/[folder-name]/[file-name]
```

### List clusters and env
```bash
which conda

conda env list
```

## 2. Understanding Cluster Resources

The cluster uses Slurm for job scheduling, and available resources include:
	•	ju and ju-interactive partitions (priority access for personal jobs).
	•	gpu and default-partitions (shared resources, may be preempted).
	•	4 GPUs available; do not request all at once.

## 3. Requesting GPU Resources (Interactive Mode)

To get an interactive session with 1 GPU, 8 CPU cores, and 32GB RAM:

srun -p ju-interactive --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=4:00:00 --pty /bin/bash

Common Parameters:
	•	-p ju-interactive → Use the interactive queue.
	•	--gres=gpu:1 → Request 1 GPU.
	•	--cpus-per-task=8 → Request 8 CPU cores.
	•	--mem=32G → Allocate 32GB RAM.
	•	--time=2:00:00 → Job runs for max 2 hours.
	•	--pty /bin/bash → Start an interactive shell.

Once inside the GPU node, **activate your environment** and run Python scripts as usual.

4. Submitting a Job with Slurm

Instead of running jobs interactively, you can submit them as batch jobs.

Step 1: Create a Job Script

Create a file called job.slurm:

#!/bin/bash
#SBATCH --job-name=qwen_vl
#SBATCH --partition=ju
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# Load Conda
source /share/apps/anaconda3/2020.11/etc/profile.d/conda.sh
conda activate /home/hq48/vlm-testing/qwen_env

# Run script
python /home/hq48/vlm-testing/analyze-video-qwen.py

Step 2: Submit the Job

mkdir -p logs  # Ensure log directory exists
sbatch job.slurm  # Submit the job

Step 3: Monitor the Job

squeue -u hq48  # Check running jobs
scancel <job_id>  # Cancel a job

Step 4: Job starting time
squeue --start  # 查看预计开始时间

5. Managing Conda Environments

Since cluster home directories may have limited space, you should create Conda environments in a specific directory.

Create and Activate a Conda Environment

conda create --prefix /home/hq48/vlm-testing/qwen_env python=3.9 -y
conda activate /home/hq48/vlm-testing/qwen_env

If conda is Not Found

First, initialize Conda:

/share/apps/anaconda3/2020.11/bin/conda init
source ~/.bashrc

If conda is still not found, add it to your PATH manually:

export PATH="/share/apps/anaconda3/2020.11/bin:$PATH"

Then, try activating the environment again:

source /share/apps/anaconda3/2020.11/etc/profile.d/conda.sh
conda activate /home/hq48/vlm-testing/qwen_env

6. Installing Required Packages

Make sure all required dependencies are installed inside your Conda environment.

Install PyTorch (GPU)

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

Install Other Dependencies

pip install transformers modelscope accelerate numpy pillow opencv-python moviepy

Check Package Installation

python -c "import torch; print(torch.__version__)"
python -c "import torchvision; print(torchvision.__version__)"

7. Running Qwen2.5-VL on the Cluster

Clone Repository

cd /home/hq48/vlm-testing
git clone https://github.com/QwenLM/Qwen2.5-VL.git
cd Qwen2.5-VL

Download Model Locally and Upload to Cluster

If the cluster cannot access Hugging Face, download the model on your local machine:

from modelscope import snapshot_download
model_dir = snapshot_download('qwen/Qwen2.5-VL-7B-Instruct')
print(f'Model downloaded to {model_dir}')

Then upload it to the cluster:

scp -r /local/path/to/Qwen2.5-VL-7B-Instruct hq48@g2-login-01:/home/hq48/vlm-testing/models/

Modify Python Code to Use Local Model

model_path = "/home/hq48/vlm-testing/models/Qwen2.5-VL-7B-Instruct"

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_path)

8. Common Issues & Fixes

Issue	Solution
conda: command not found	Run /share/apps/anaconda3/2020.11/bin/conda init and source ~/.bashrc
ModuleNotFoundError: No module named 'torchvision'	Run pip install torchvision in the correct Conda environment
nvidia-smi: command not found	You are on a login node; use srun to request a GPU
requests.exceptions.ReadTimeout: Hugging Face timeout	Download the model locally and upload it manually
OSError: We couldn't connect to 'huggingface.co'	Set local_files_only=True when loading the model

9. Running Jupyter Notebook on the Cluster

Step 1: Start Jupyter on the Cluster

jupyter notebook --no-browser --port=8888 --ip=0.0.0.0

Step 2: Connect from Your Local Machine

On your local terminal:

ssh -L 8888:localhost:8888 hq48@g2-login-01

Then open http://localhost:8888 in your browser.

10. Summary

Key Commands

Task	Command
Login to cluster	ssh hq48@g2-login-01
Request GPU (interactive)	srun -p ju-interactive --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=2:00:00 --pty /bin/bash
Submit job	sbatch job.slurm
Check job status	squeue -u hq48
Cancel job	scancel <job_id>
Activate Conda environment	conda activate /home/hq48/vlm-testing/qwen_env
Install packages	pip install transformers modelscope
Check GPU usage	nvidia-smi
Run ollama model	./ollama-linux-amd64 serve