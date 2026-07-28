#!/bin/bash
#SBATCH --job-name=davis
#SBATCH --output=./logs/%x_%j.log    # x: job name , j: job ID, a: array task ID
#SBATCH --error=./logs/%x_%j.err
#SBATCH --mem=45G
#SBATCH --time=10:00:00
#SBATCH --partition=cbio-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A40:1               # A40 (48Go) , P100 (16Go, fastest), T4 (16Go, especially designed for inference), K80
##SBATCH --cpus-per-task=1
#SBATCH --mail-type=END                 # Send email at job completion
#SBATCH --mail-user=sioban.nieradzik-kozic@etu.minesparis.psl.eu


RUN_DIR="/cluster/CBIO/home/snieradzikkozic/ADiT"
RUN_SCRIPT="./adit/train.py"
DATA="davis"

COORD_ENCODING="onehot"     # onehot , rbf , null

CKPT_EPOCH=33 #$((SLURM_ARRAY_TASK_ID * 32))
CKPT_PATH="ckpts/${COORD_ENCODING}/epoch_0${CKPT_EPOCH}.ckpt"
# CKPT_PATH="outputs/2026-07-20/denoising_S_scale0_5_12-49-46/checkpoints/epoch_033.ckpt"

# Protein - Ligand : "lba"
# Drug - Target : "davis"
# Protein - Protein : "skempi"

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
source ~/.bashrc
conda activate adit
# ---- Run it...
cd "$RUN_DIR"
set -x
echo 'Launching script'

srun python "$RUN_SCRIPT" \
    experiment=davis_S \
    ++task_name=davis_S_$COORD_ENCODING \
    ++model.net.token_coord_encoder=$COORD_ENCODING \
    ckpt_path=$CKPT_PATH \
    ++trainer.devices=$SLURM_NTASKS_PER_NODE \
    logger=many_loggers \
    ++logger.wandb.name=davis_${COORD_ENCODING}_epoch_$CKPT_EPOCH \
    ++logger.wandb.id=davis_${COORD_ENCODING}_${SLURM_JOB_ID} \
    ++logger.wandb.project=davis
