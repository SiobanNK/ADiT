#!/bin/bash
<<<<<<< HEAD
#SBATCH --job-name=skempi
#SBATCH --output=./logs/%x_%j_%a.log    # x: job name , j: job ID, a: array task ID
#SBATCH --error=./logs/%x_%j_%a.err
#SBATCH --array=0
#SBATCH --mem=40G
#SBATCH --time=05:00:00
=======
#SBATCH --job-name=skempi_rbf
#SBATCH --output=./logs/%x_%a_%j.log    # x: job name , j: job ID, a: array task ID
#SBATCH --error=./logs/%x_%a_%j.err
#SBATCH --array=1-2
#SBATCH --mem=45G
#SBATCH --time=07:00:00
>>>>>>> tokenpos
#SBATCH --partition=cbio-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A40:1               # A40 (48Go) , P100 (16Go, fastest), T4 (16Go, especially designed for inference), K80
<<<<<<< HEAD
#SBATCH --cpus-per-task=1
=======
#SBATCH --cpus-per-task=4              # CPU cores per process (default 1, typically 4 or 5 - do not use more to let space for other people)
>>>>>>> tokenpos
#SBATCH --mail-type=END                 # Send email at job completion
#SBATCH --mail-user=sioban.nieradzik-kozic@etu.minesparis.psl.eu


<<<<<<< HEAD
# Protein - Ligand : "lba"
# Drug - Target : "davis"
# Protein - Protein : "skempi"
# Antibody-Antigen : "her2"


RUN_DIR="$SCRATCH/ADiT"
RUN_SCRIPT="./adit/train.py"
SPLIT="$SLURM_ARRAY_TASK_ID"
DATA="$SLURM_JOB_NAME"
ADIT_SIZE="S"
=======
RUN_DIR="/cluster/CBIO/home/snieradzikkozic/ADiT"
RUN_SCRIPT="./adit/train.py"
DATA="skempi"
SPLIT="$SLURM_ARRAY_TASK_ID"

COORD_ENCODING="rbf"     # onehot , rbf , null
D_MAX=50

CKPT_EPOCH=32 #$((SLURM_ARRAY_TASK_ID * 32))
CKPT_PATH="ckpts/${COORD_ENCODING}/epoch_0${CKPT_EPOCH}.ckpt"
# CKPT_PATH="outputs/2026-07-20/denoising_S_scale0_5_12-49-46/checkpoints/epoch_033.ckpt"

# Protein - Ligand : "lba"
# Drug - Target : "davis"
# Protein - Protein : "skempi"
>>>>>>> tokenpos

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
<<<<<<< HEAD
echo 'Launching script'


srun python "$RUN_SCRIPT" \
    experiment=${DATA}_$ADIT_SIZE \
    data.dataset.test_split=split_$SPLIT \
    task_name=${DATA}${SPLIT}_${ADIT_SIZE} \
    ++trainer.devices=$SLURM_NTASKS_PER_NODE \
    ++data.num_workers=$SLURM_CPUS_PER_TASK \
    ckpt_path=ckpts/adit_$ADIT_SIZE.ckpt \
    logger=many_loggers \
    ++logger.wandb.name=${DATA}${SPLIT}_$ADIT_SIZE \
    ++logger.wandb.id=${DATA}_${SLURM_JOB_ID}_$SLURM_ARRAY_TASK_ID \
    ++logger.wandb.project=${DATA} \
    ++logger.wandb.offline=False \
    # ++trainer.reload_dataloaders_every_n_epochs=1 \
    # ++trainer.use_distributed_sampler=False \
    # ++data.max_pair_budget=250000 \
    # ++data.max_total_atoms=15000 \


# trainer.devices : S : 2, M : 4, L : 4
=======
set -x
echo 'Launching script'

srun python "$RUN_SCRIPT" \
    experiment=${DATA}_S \
    data.dataset.test_split=split_$SPLIT \
    ++task_name=${DATA}${SPLIT}_S_${COORD_ENCODING}_epoch_$CKPT_EPOCH \
    ++model.net.token_coord_encoder=$COORD_ENCODING \
    ++model.net.d_max=$D_MAX \
    ckpt_path=$CKPT_PATH \
    logger=many_loggers \
    ++logger.wandb.name=${DATA}${SPLIT}_${COORD_ENCODING}_epoch_$CKPT_EPOCH \
    ++logger.wandb.id=${DATA}${SPLIT}_${COORD_ENCODING}_${SLURM_JOB_ID} \
    ++logger.wandb.project=${DATA}
>>>>>>> tokenpos
