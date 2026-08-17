#!/bin/bash
#SBATCH --job-name=skempi_rbf
#SBATCH --output=./logs/%x_%a_%j.log    # x: job name , j: job ID, a: array task ID
#SBATCH --error=./logs/%x_%a_%j.err
#SBATCH --array=0-2
#SBATCH --mem=45G
#SBATCH --time=07:00:00
#SBATCH --partition=cbio-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A40:1               # A40 (48Go) , P100 (16Go, fastest), T4 (16Go, especially designed for inference), K80
#SBATCH --cpus-per-task=4              # CPU cores per process (default 1, typically 4 or 5 - do not use more to let space for other people)
#SBATCH --mail-type=END                 # Send email at job completion
#SBATCH --mail-user=sioban.nieradzik-kozic@etu.minesparis.psl.eu


RUN_DIR="/cluster/CBIO/home/snieradzikkozic/ADiT"
RUN_SCRIPT="./adit/train.py"
DATA="skempi"
SPLIT="$SLURM_ARRAY_TASK_ID"

COORD_ENCODING="rbf"     # onehot , rbf , null
D_MAX=2.2

CKPT_EPOCH=32 #$((SLURM_ARRAY_TASK_ID * 32))
CKPT_PATH="ckpts/${COORD_ENCODING}/dist_max_22/epoch_0${CKPT_EPOCH}.ckpt"
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
    experiment=${DATA}_S \
    data.dataset.test_split=split_$SPLIT \
    ++task_name=${DATA}${SPLIT}_S_${COORD_ENCODING}_epoch_$CKPT_EPOCH \
    ++model.net.token_coord_encoder=$COORD_ENCODING \
    ++model.net.relative_position_d_max=$D_MAX \
    ckpt_path=$CKPT_PATH \
    trainer.devices=$SLURM_NTASKS_PER_NODE \
    logger=many_loggers \
    ++logger.wandb.name=${DATA}${SPLIT}_${COORD_ENCODING}_d${D_MAX}_epoch_$CKPT_EPOCH \
    ++logger.wandb.id=${DATA}${SPLIT}_${COORD_ENCODING}_${SLURM_JOB_ID} \
    ++logger.wandb.project=${DATA}
