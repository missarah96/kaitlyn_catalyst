#!/bin/bash
#SBATCH --job-name=wandb_sweep
#SBATCH --output=sweep_output_%j.txt
#SBATCH --error=sweep_error_%j.txt
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1           # Remove if not using GPU
#SBATCH --partition=gpu_partition  # Change to your actual partition (e.g., gpu, compute)

# Load conda and activate environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate speciesnet

# Navigate to the project directory
cd ~/Desktop/Kaitlyn_Catalyst/ct_classifier/speciesnet

# Run the sweep agent
# wandb agent catalyst_dsi/Species-Classification/b9a8yru8
