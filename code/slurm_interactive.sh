#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --qos=high
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/home/%u/logs/%j.out
#SBATCH --no-container-entrypoint
#SBATCH --container-mounts=/data/pathology:/data/pathology,/data/pa_cpgarchive:/data/pa_cpgarchive
#SBATCH --container-image="dockerdex.umcn.nl:5005/pierpaolov/pathology-pipeline:2.0"

mkdir -p /home/${USER}/logs

jupyter lab --ip=0.0.0.0 --port=5244 --no-browser
