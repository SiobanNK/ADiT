#!/bin/bash
#SBATCH --job-name=lba30_rbf
#SBATCH --output=./logs/%x30_%j.log    # x: job name , j: job ID, a: array task ID
#SBATCH --error=./logs/%x30_%j.err
#SBATCH --array=42
#SBATCH --mem=45G
#SBATCH --time=20:00:00
#SBATCH --partition=cbio-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A40:1               # A40 (48Go) , P100 (16Go, fastest), T4 (16Go, especially designed for inference), K80
##SBATCH --cpus-per-task=8
#SBATCH --mail-type=END                 # Send email at job completion
#SBATCH --mail-user=sioban.nieradzik-kozic@etu.minesparis.psl.eu

# Protein - Ligand : "lba"
# Drug - Target : "davis"
# Protein - Protein : "skempi"

RUN_DIR="/cluster/CBIO/home/snieradzikkozic/ADiT"
RUN_SCRIPT="./adit/train.py"
DATA="lba"
SIZE="S"
IDENTITY=60 # "$SLURM_ARRAY_TASK_ID"
SEED="$SLURM_ARRAY_TASK_ID"
if [[ "$IDENTITY" -eq 30 ]]; then
    DROPOUT=0.0
    EPOCHS=10 # 3
else    # IDENTITY=60
    DROPOUT=0.5
    EPOCHS=99
fi  # cf article

COORD_ENCODING="null"     # onehot , rbf , null
D_MAX=2.2

CKPT_EPOCH=32 #$((SLURM_ARRAY_TASK_ID * 32))
# CKPT_PATH="ckpts/rbf/dist_max_22/epoch_0${CKPT_EPOCH}.ckpt"
CKPT_PATH="ckpts/${COORD_ENCODING}/epoch_0${CKPT_EPOCH}.ckpt"

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
    seed=$SEED \
    data.batch_size=60  \
    data.dataset.split_identity_threshold=identity_$IDENTITY \
    task_name=${DATA}${IDENTITY}_seed${SEED}_${SIZE}_${COORD_ENCODING}_epoch_$CKPT_EPOCH \
    +model.net.token_coord_encoder=$COORD_ENCODING \
    +model.net.relative_position_d_max=$D_MAX \
    model.net.dropout=$DROPOUT \
    trainer.max_epochs=$EPOCHS \
    ckpt_path=$CKPT_PATH \
    trainer.devices=$SLURM_NTASKS_PER_NODE \
    logger=many_loggers \
    logger.wandb.name=${DATA}${IDENTITY}_${COORD_ENCODING}_d${D_MAX}_epoch_$CKPT_EPOCH \
    logger.wandb.id=${DATA}${IDENTITY}_${COORD_ENCODING}_${SLURM_JOB_ID} \
    ++logger.wandb.project=$DATA \
    test=true
