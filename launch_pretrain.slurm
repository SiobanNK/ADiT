#!/bin/bash
#SBATCH --job-name=pretrain
#SBATCH --output=./logs/%x_%j.log    # x: job name , j: job ID, a: array task ID
#SBATCH --error=./logs/%x_%j.err
#SBATCH --mem=40G
#SBATCH --time=14-00:00:00
#SBATCH --partition=cbio-gpu
#SBATCH --gres=gpu:A100:1               # A40 (48Go) , P100 (16Go, fastest), T4 (16Go, especially designed for inference), K80
#SBATCH --nodes=1
##SBATCH --cpus-per-task=1
#SBATCH --mail-type=END                 # Send email at job completion
#SBATCH --mail-user=sioban.nieradzik-kozic@etu.minesparis.psl.eu

RUN_DIR="/cluster/CBIO/home/snieradzikkozic/ADiT"
RUN_SCRIPT="./adit/train.py"
ADIT_SIZE="S"       # ADAPT: L, S or AF3

# ---- Welcome...
echo '-------------------------------------'
echo "[$0] >>> Start : $(date '+%d/%m/%Y-%H:%M:%S')"
echo '-------------------------------------'
echo "Job id : $SLURM_JOB_ID"
echo "Job name : $SLURM_JOB_NAME"
echo "Job node list : $SLURM_JOB_NODELIST"
echo '--------------------------------------'
echo "Script : $RUN_SCRIPT"
echo "Run in : $RUN_DIR"
echo '--------- --------------------------'
# ---- Module
module purge
# ---- Run it...
source /cluster/CBIO/home/snieradzikkozic/miniconda3/etc/profile.d/conda.sh
conda activate adit
cd "$RUN_DIR"
echo 'Launching script'

export NUMEXPR_MAX_THREADS=8

# take test_split=split_0 as an example
srun python "$RUN_SCRIPT" experiment=new_denoise_${ADIT_SIZE}_pdb_fixed_0_5
