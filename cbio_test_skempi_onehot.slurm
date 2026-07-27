#!/bin/bash
#SBATCH --job-name=skempi_onehot
#SBATCH --output=./logs/%x_%a_%j.log    # x: job name , j: job ID, a: array task ID
#SBATCH --error=./logs/%x_%a_%j.err
#SBATCH --array=2
#SBATCH --mem=15G
#SBATCH --time=00:10:00
#SBATCH --partition=cbio-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:P100:1               # A40 (48Go) , P100 (16Go, fastest), T4 (16Go, especially designed for inference), K80
##SBATCH --cpus-per-task=1
#SBATCH --mail-type=END                 # Send email at job completion
#SBATCH --mail-user=sioban.nieradzik-kozic@etu.minesparis.psl.eu


RUN_DIR="/cluster/CBIO/home/snieradzikkozic/ADiT"
RUN_SCRIPT="./adit/test.py"
DATA="skempi"
SPLIT="$SLURM_ARRAY_TASK_ID"

COORD_ENCODING="onehot"     # onehot , rbf , null

CKPT_EPOCH=95 #$((SLURM_ARRAY_TASK_ID * 32))
CKPT_PATH="/outputs/2026-07-27/skempi2_onehot_00-55-00/checkpoints/last.ckpt"

SAVE_FILE="outputs/onehot/epoch_${EPOCH}/result_split_$SPLIT"

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
    experiment=skempi_S_ft_onehot \
    ++data.batch_size=1 \
    ++trainer.devices=1 \
    ++data.dataset.test_split=split_$SPLIT \
    ++model.save_file=$SAVE_FILE \
    ++task_name=skempi${SPLIT}_S_onehot \
    ckpt_path=$CKPT_PATH \
