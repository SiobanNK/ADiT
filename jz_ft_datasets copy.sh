#!/bin/bash
#SBATCH --job-name=ckpts_download
#SBATCH --partition=prepost
#SBATCH --time=02:00:00
#SBATCH --output=%x.%j.out

module load python
export HF_HOME=$SCRATCH/hf_cache
export TARGET=$SCRATCH/ADiT/ckpts

hf download VectorShi/ADiT_dataset --repo-type model --local-dir $TARGET
