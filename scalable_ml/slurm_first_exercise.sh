#!/bin/bash
#SBATCH --partition=A40devel
#SBATCH --time=0:15:00
#SBATCH --gpus=1
#SBATCH --ntasks=1

# Note that this is script shall give you an idea how to run your code on the HPC cluster Bender
module load Miniforge3
source ~/.bashrc
conda activate scalable_ml_env
# The next two lines have to be adapted to your specific file paths (your username is not aruettg1)
export PYTHONPATH="${PYTHONPATH}:/home/aruettg1/.conda/envs/scalable_ml_env/lib/python3.9/site-packages/scalable_ml/"
python /home/aruettg1/scalable_ml/tools/first_exercise_pytorch_with_fmnist.py
