#!/bin/bash
#SBATCH --job-name=lba
#SBATCH --output=./logs/%x%a_%j.log    # x: job name , j: job ID, a: array task ID
#SBATCH --error=./logs/%x%a_%j.err
#SBATCH --array=60
#SBATCH --mem=15G
#SBATCH --time=10:00:00
#SBATCH --partition=cbio-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:P100:2               # A40 (48Go) , P100 (16Go, fastest), T4 (16Go, especially designed for inference), K80
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=END                 # Send email at job completion
#SBATCH --mail-user=sioban.nieradzik-kozic@etu.minesparis.psl.eu

# Protein - Ligand : "lba"
# Drug - Target : "davis"
# Protein - Protein : "skempi"

RUN_DIR="/cluster/CBIO/home/snieradzikkozic/ADiT"
RUN_SCRIPT="./adit/train.py"
DATA="lba"
SIZE="S"
IDENTITY=$SLURM_ARRAY_TASK_ID
SEED=42 #"$SLURM_ARRAY_TASK_ID"
if [[ "$IDENTITY" -eq 30 ]]; then
    DROPOUT=0.0
    EPOCHS=10 # 3
else    # IDENTITY=60
    DROPOUT=0.5
    EPOCHS=99
fi  # cf article

COORD_ENCODING="null"
# CKPT_PATH="ckpts/rbf/dist_max_22/epoch_0${CKPT_EPOCH}.ckpt"
CKPT_PATH="ckpts/${COORD_ENCODING}/epoch_032.ckpt"

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
    test=true \
    data.dataset.split_identity_threshold=identity_$IDENTITY \
    task_name=${DATA}${IDENTITY}_${SIZE} \
    model.net.dropout=$DROPOUT \
    ckpt_path=$CKPT_PATH \
    +data.drop_last=true \
    trainer.max_epochs=$EPOCHS \
    trainer.devices=$SLURM_NTASKS_PER_NODE \
    logger=many_loggers \
    logger.wandb.name=${DATA}${IDENTITY} \
    logger.wandb.id=${DATA}${IDENTITY}_${SLURM_JOB_ID} \
    ++logger.wandb.project="Profiling"
