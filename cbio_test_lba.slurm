#!/bin/bash
#SBATCH --job-name=lba_test
#SBATCH --output=./logs/%x_%j.log    # x: job name , j: job ID, a: array task ID
#SBATCH --error=./logs/%x_%j.err
#SBATCH --mem=15G
#SBATCH --time=01:00:00
#SBATCH --partition=cbio-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:P100:1             # A40 (48Go) , P100 (16Go, fastest), T4 (16Go, especially designed for inference), K80
##SBATCH --cpus-per-task=1
#SBATCH --mail-type=END                 # Send email at job completion
#SBATCH --mail-user=sioban.nieradzik-kozic@etu.minesparis.psl.eu


RUN_DIR="/cluster/CBIO/home/snieradzikkozic/ADiT"
RUN_SCRIPT="./adit/test.py"
DATA="lba"
IDENTITY=30
SIZE="S"

COORD_ENCODING="rbf"     # onehot , rbf , null
D_MAX=2.2

EPOCH=32 #$((SLURM_ARRAY_TASK_ID * 32))
# CKPT_PATH="outputs/${COORD_ENCODING}/epoch_${EPOCH}/${DATA}${IDENTITY}_${SIZE}_d22/checkpoints/last.ckpt"
CKPT_PATH="outputs/${COORD_ENCODING}/epoch_${EPOCH}/${DATA}${IDENTITY}_${SIZE}_d22/checkpoints/epoch_002.ckpt"

SAVE_FILE="outputs/${COORD_ENCODING}/epoch_${EPOCH}/result_${DATA}${IDENTITY}_${SIZE}_d22.pkl"

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
    experiment=${DATA}_${SIZE} \
    ++model.net.token_coord_encoder=$COORD_ENCODING \
    ++data.dataset.split_identity_threshold=identity_$IDENTITY \
    ++data.batch_size=1 \
    ++trainer.devices=1 \
    ++model.save_file=$SAVE_FILE \
    ++task_name=test_${DATA}${IDENTITY}_${SIZE}_$COORD_ENCODING \
    ckpt_path=$CKPT_PATH \
