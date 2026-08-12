#!/bin/bash
#SBATCH --job-name=pretrain_gat
#SBATCH --output=./logs/%x_%j.log    # x: job name , j: job ID, a: array task ID
#SBATCH --error=./logs/%x_%j.err
#SBATCH --array=15
#SBATCH --mem=45G
#SBATCH --time=02:00:00
#SBATCH --partition=cbio-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A40:1               # A40 (48Go) , P100 (16Go, fastest), T4 (16Go, especially designed for inference), K80
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=END                 # Send email at job completion
#SBATCH --mail-user=sioban.nieradzik-kozic@etu.minesparis.psl.eu




RUN_DIR="/cluster/CBIO/home/snieradzikkozic/ADiT"
RUN_SCRIPT="./adit/train.py"
ADIT_SIZE="XS"

EPOCHS=3

# Définition des 4 options pour ATOM et 4 options pour TOKEN
ATOM_LIST=(
  "[False,False]" "[True,False]" "[False,True]" "[True,True]"
)
TOKEN_LIST=(
  "[False,False]" "[True,False]" "[False,True]" "[True,True]"
)

# Conversion de SLURM_ARRAY_TASK_ID (0 à 15) en 2 index (0 à 3)
ATOM_IDX=$(( SLURM_ARRAY_TASK_ID / 4 ))
TOKEN_IDX=$(( SLURM_ARRAY_TASK_ID % 4 ))

ATOM_PARAM=${ATOM_LIST[$ATOM_IDX]}
TOKEN_PARAM=${TOKEN_LIST[$TOKEN_IDX]}

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
    experiment=new_denoise_${ADIT_SIZE}_pdb_fixed_0_5 \
    +model.net.atom_neighbour_radius="5.0" \
    +model.net.token_neighbour_radius="10.0" \
    +model.net.atom_gat_positions=$ATOM_PARAM    \
    +model.net.token_gat_positions=$TOKEN_PARAM \
    trainer.devices=$SLURM_NTASKS_PER_NODE \
    trainer.min_epochs=$EPOCHS \
    trainer.max_epochs=$EPOCHS \
    task_name=gat${SLURM_ARRAY_TASK_ID}_denoising_${ADIT_SIZE} \
    logger=many_loggers \
    ++logger.wandb.name=gat${SLURM_ARRAY_TASK_ID}_pretrain\
    ++logger.wandb.id=gat${SLURM_ARRAY_TASK_ID}_pretrain_$SLURM_JOB_ID \
    ++logger.wandb.project="pretrain"
