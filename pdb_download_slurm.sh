#!/bin/bash
#SBATCH --job-name=pdb_download
#SBATCH --time=72:00:00
#SBATCH --output=./dataset/pdb/slurm_download.log
#SBATCH --error=./dataset/pdb/slurm_download.err
#SBATCH --partition=cbio-gpu
#SBATCH --cpus-per-task=2
#SBATCH --mail-type=END                 # Send email at job completion
#SBATCH --mail-user=sioban.nieradzik-kozic@etu.minesparis.psl.eu


wget -r -np -nH \
  --cut-dirs=5 \
  --accept "*.cif.gz" \
  --tries=5 \
  --wait=1 \
  --continue \
  -P ./dataset/pdb \
  https://files-beta.rcsb.org/pub/wwpdb/pdb/data/entries/
