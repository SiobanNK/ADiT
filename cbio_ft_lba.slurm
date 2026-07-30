#!/bin/bash
#SBATCH --job-name=lba
#SBATCH --output=./logs/%x%a_%j.log    # x: job name , j: job ID, a: array task ID
#SBATCH --error=./logs/%x%a_%j.err
#SBATCH --array=30
#SBATCH --mem=15G
#SBATCH --time=01:00:00
#SBATCH --partition=cbio-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:P100:1               # A40 (48Go) , P100 (16Go, fastest), T4 (16Go, especially designed for inference), K80
##SBATCH --cpus-per-task=1
#SBATCH --mail-type=END                 # Send email at job completion
#SBATCH --mail-user=sioban.nieradzik-kozic@etu.minesparis.psl.eu

# Protein - Ligand : "lba"
# Drug - Target : "davis"
# Protein - Protein : "skempi"

RUN_DIR="/cluster/CBIO/home/snieradzikkozic/ADiT"
RUN_SCRIPT="./adit/train.py"
DATA="lba"
IDENTITY=$SLURM_ARRAY_TASK_ID
if [[ "$IDENTITY" -eq 60 ]]; then
    DROPOUT=0.5
    EPOCHS=3 # total: 3 epochs
else    # IDENTITY=30
    DROPOUT=0.0
    EPOCHs=99
fi  # cf articlefi  # cf articleSIZE="S"

COORD_ENCODING="onehot"     # onehot , rbf , null

CKPT_EPOCH=33 #$((SLURM_ARRAY_TASK_ID * 32))
CKPT_PATH="ckpts/onehot/epoch_0${CKPT_EPOCH}.ckpt"

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

python "$RUN_SCRIPT" \
    experiment=${DATA}_$SIZE \
    ++data.dataset.split_identity_threshold=identity_$IDENTITY \
    ++task_name=${DATA}${IDENTITY}_${SIZE}_${COORD_ENCODING}_epoch_$CKPT_EPOCH \
    ++model.net.token_coord_encoder=$COORD_ENCODING \
    ++model.net.dropout=$DROPOUT \
    ++trainer.max_epochs=$EPOCHS \
    ckpt_path=$CKPT_PATH \
    ++trainer.devices=$SLURM_NTASKS_PER_NODE \
    logger=many_loggers \
    ++logger.wandb.name=${DATA}${IDENTITY}_${COORD_ENCODING}_epoch_$CKPT_EPOCH \
    ++logger.wandb.id=${DATA}_${COORD_ENCODING}_${SLURM_JOB_ID} \
    ++logger.wandb.project=$DATA
