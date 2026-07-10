#!/bin/bash
#SBATCH --job-name=davis
#SBATCH --output=./logs/%x_%j.log    # x: job name , j: job ID, a: array task ID
#SBATCH --error=./logs/%x_%j.err
#SBATCH --mem=15G
#SBATCH --time=01:00:00
#SBATCH --partition=cbio-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:P100:2               # A40 (48Go) , P100 (16Go, fastest), T4 (16Go, especially designed for inference), K80
##SBATCH --cpus-per-task=1
#SBATCH --mail-type=END                 # Send email at job completion
#SBATCH --mail-user=sioban.nieradzik-kozic@etu.minesparis.psl.eu


RUN_DIR="/cluster/CBIO/home/snieradzikkozic/ADiT"
RUN_SCRIPT="./adit/train.py"
DATA="$SLURM_JOB_NAME"
ADIT_SIZE="S"

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
source ~/.bashrc
conda deactivate
conda activate $WORK/envs/adit
# ---- Run it...
cd "$RUN_DIR"
set -x
echo 'Launching script'

srun python "$RUN_SCRIPT" \
    experiment=${DATA}_$ADIT_SIZE \
    data=$DATA \
    ++trainer.devices=2 \
    ++data.batch_size=8 \
    ++data.dataset.split_seed=seed_1 \
    ckpt_path=ckpts/adit_$ADIT_SIZE.ckpt \
    logger=wandb \
    ++logger.wandb.name=${DATA}_$ADIT_SIZE \
    ++logger.wandb.id=${DATA}_$ADIT_SIZE_$SLURM_JOB_ID

# trainer.devices : S : 2, M : 4, L : 4
