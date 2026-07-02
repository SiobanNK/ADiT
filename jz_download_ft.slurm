#!/bin/bash
#SBATCH --job-name=adit_download
#SBATCH --partition=prepost
#SBATCH --time=02:00:00
#SBATCH --output=%x.%j.out

module load python
export HF_HOME=$SCRATCH/hf_cache
export TARGET=$SCRATCH/ADiT/dataset
mkdir -p $TARGET

hf download VectorShi/ADiT_dataset --repo-type dataset --local-dir $TARGET
