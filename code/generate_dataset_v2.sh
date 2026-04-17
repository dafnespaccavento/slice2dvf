#!/bin/bash
#SBATCH -A NAISS2026-3-68
#SBATCH -p alvis
#SBATCH -t 12:00:00
#SBATCH --cpus-per-task=4
#SBATCH -C NOGPU
#SBATCH -J generate_dvf_2
#SBATCH -o /mimer/NOBACKUP/groups/caim1/dafne/logs/generate_%j.out
#SBATCH -e /mimer/NOBACKUP/groups/caim1/dafne/logs/generate_%j.err

module load Python/3.10.4-GCCcore-11.3.0
source /mimer/NOBACKUP/groups/caim1/dafne/venv/bin/activate
python /mimer/NOBACKUP/groups/caim1/dafne/code/generate_dataset_v2.py
