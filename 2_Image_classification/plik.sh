#!/bin/bash
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=10gb
#SBATCH --partition lem-gpu
#SBATCH --job-name=GPU_test
#SBATCH --mail-user=275340@student.pwr.edu.pl
#SBATCH --gres=gpu:hopper:1
#SBATCH --account=hpc-jarpaw-1791296019
#SBATCH --output=log_%j


apptainer exec --nv \
  --bind "$HOME/python-packages:/opt/python-packages" \
  /lustre/software-data/container-images/pytorch_latest.sif \
  bash -c "export PYTHONPATH=/opt/python-packages:$PYTHONPATH; python ./Crystal.py"


#copying log into results folder
TARGET_DIR="ResultsCrystal"

cp "log_${SLURM_JOB_ID}" "$TARGET_DIR/"
