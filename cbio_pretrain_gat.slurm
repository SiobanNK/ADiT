#!/bin/bash
#SBATCH --job-name=pretrain_gat
#SBATCH --output=./logs/%x_%j.log    # x: job name , j: job ID, a: array task ID
#SBATCH --error=./logs/%x_%j.err
## SBATCH --array=0
#SBATCH --mem=45G
#SBATCH --time=10:00:00
#SBATCH --partition=cbio-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A40:1               # A40 (48Go) , P100 (16Go, fastest), T4 (16Go, especially designed for inference), K80
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=END                 # Send email at job completion
#SBATCH --mail-user=sioban.nieradzik-kozic@etu.minesparis.psl.eu


RUN_DIR="/cluster/CBIO/home/snieradzikkozic/ADiT"
RUN_SCRIPT="./adit/train.py"
ADIT_SIZE="XS"
GNN_TYPE='gat'  # gcn / gat
CKPT="outputs/gat_Cbeta_denoising_XS/checkpoints/epoch_005.ckpt"
CKPT_WANDB_ID="gat_pretrain_1329674"

EPOCHS=12

# # Nombre à convertir (0-64), sur 8 bits ici -> adapte NBITS si besoin
# NUM=$SLURM_ARRAY_TASK_ID
# NBITS=8

# # Conversion en binaire, complété par des zéros à gauche
# BIN=$(printf '%0*d' "$NBITS" "$(echo "obase=2;$NUM" | bc)")

# # Construction de la liste Hydra-compatible [False, True, ...]
# LIST="["
# for (( i=0; i<${#BIN}; i++ )); do
#     bit=${BIN:$i:1}
#     if [ "$bit" = "1" ]; then
#         LIST+="True"
#     else
#         LIST+="False"
#     fi
#     if [ $i -lt $((${#BIN}-1)) ]; then
#         LIST+=","
#     fi
# done
# LIST+="]"

echo "Nombre: $NUM -> Binaire: $BIN -> Liste: $LIST"
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
    +model/gnn=$GNN_TYPE \
    ckpt=$CKPT \
    trainer.devices=$SLURM_NTASKS_PER_NODE \
    trainer.min_epochs=$EPOCHS \
    trainer.max_epochs=$EPOCHS \
    data.num_workers=$SLURM_CPUS_PER_TASK \
    task_name=${GNN_TYPE}_Cbeta_denoising_${ADIT_SIZE}_end \
    logger=many_loggers \
    ++logger.wandb.name=${GNN_TYPE}_CB_${ADIT_SIZE}\
    ++logger.wandb.id=$CKPT_WANDB_ID \
    ++logger.wandb.project="pretrain"

#  ${GNN_TYPE}_pretrain_$CKPT_SLURM_JOB_ID
