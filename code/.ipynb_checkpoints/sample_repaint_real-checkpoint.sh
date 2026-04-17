#!/bin/bash
#SBATCH --job-name=repaint_real
#SBATCH --account=NAISS2026-3-68
#SBATCH --partition=alvis
#SBATCH --time=0-06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:A100:1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

module purge
module load Python/3.10.4-GCCcore-11.3.0
source /mimer/NOBACKUP/groups/caim1/dafne/venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

python -u /mimer/NOBACKUP/groups/caim1/dafne/code/sample_repaint_real_pretrainedweights.py
