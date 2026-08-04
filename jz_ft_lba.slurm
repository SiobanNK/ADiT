#!/bin/bash
#SBATCH --job-name=lba_null
#SBATCH --output=./logs/%x%a_%j.log    # x: job name , j: job ID, a: array task ID
#SBATCH --error=./logs/%x%a_%j.err
#SBATCH --array=30,60
#SBATCH --mail-type=END                 # Send email at job completion
#SBATCH --mail-user=sioban.nieradzik-kozic@etu.minesparis.psl.eu

#SBATCH --account=hhy@h100      # echo $IDRPROJ
#SBATCH -C h100                 # partition gpu_p6 : noeud quadri-GPU H100 80 Go
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
# Le nombre de CPU par tache doit etre adapte en fonction de la partition utilisee. Sachant
# qu'ici on ne reserve qu'un seul GPU (soit 1/4 des GPU du noeud), l'ideal est de reserver 1/4
# des CPU du noeud pour la seule tache :
#SBATCH --cpus-per-task=24           # nombre de CPU par tache pour gpu_p6 (1/4 des CPU du noeud 4-GPU H100)
#SBATCH --hint=nomultithread        # réservation des cœurs physiques (pas d'hyperthreading)
#SBATCH --time=10:00:00
#SBATCH --qos=qos_gpu_h100-t3    # qos_cpu-t3 (default, 20h), qos_cpu-t4 (100h), qos_cpu-dev (2h)


# Protein - Ligand : "lba"
# Drug - Target : "davis"
# Protein - Protein : "skempi"

RUN_DIR="/cluster/CBIO/home/snieradzikkozic/ADiT"
RUN_SCRIPT="./adit/train.py"
DATA="lba"
SIZE="S"
IDENTITY="$SLURM_ARRAY_TASK_ID"
if [[ "$IDENTITY" -eq 30 ]]; then
    DROPOUT=0.0
    EPOCHS=3 # total: 3 epochs
else    # IDENTITY=60
    DROPOUT=0.5
    EPOCHS=99
fi  # cf articlefi  # cf articleSIZE="S"

COORD_ENCODING="null"     # onehot , rbf , null

CKPT_EPOCH=32 #$((SLURM_ARRAY_TASK_ID * 32))
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
