#!/bin/bash
#SBATCH --job-name=cfg_finetune
#SBATCH --account=NAISS2026-3-68
#SBATCH --partition=alvis
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:A100:1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

module purge
module load Python/3.10.4-GCCcore-11.3.0
source /mimer/NOBACKUP/groups/caim1/dafne/venv/bin/activate

# Fix PyTorch memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Match CPU threads to SLURM allocation
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Run CFG fine-tuning (unbuffered output!)
python -u /mimer/NOBACKUP/groups/caim1/dafne/code/train_cfg.py
