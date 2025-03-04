# Scalable_ML on HPC cluster Bender

Setting up and executing Python code on an HPC cluster like Bender requires some special steps, 
which are briefly outlined in this readme. The topic is discussed in detail on exercise sheet 4. 
If you are interested, you are welcome to try to execute your code on Bender independently. The following instructions 
should provide some hints (but there might be better solutions). I am also not sure if the installation
has to be performed on the compute node (first command) but keep in mind that the login node does not have a
GPU so that PyTorch without CUDA support might be installed then.

## Setting up a conda environment on Bender and installing the packages

srun --pty --gpus=1 /bin/bash

module load Miniforge3

module load CUDA/12.4.0

conda create -n scalable_ml_env python=3.9

conda init

conda activate scalable_ml_env

pip install .

## Running the code with a jobscript such as "slurm_first_exercis.sh"
### Submit jobscript using the sbatch command from Slurm
sbatch slurm_first_exercise.sh
