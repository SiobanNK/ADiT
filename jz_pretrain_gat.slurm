#!/bin/bash
#SBATCH --job-name=pretrain_gat
#SBATCH --output=./logs/%x%a_%j.log    # x: job name , j: job ID, a: array task ID
#SBATCH --error=./logs/%x%a_%j.err     # ${SLURM_ARRAY_TASK_ID}
##SBATCH --array=0,5,9,10,11,13,14,15
#SBATCH --mail-type=END                 # Send email at job completion
#SBATCH --mail-user=sioban.nieradzik-kozic@etu.minesparis.psl.eu

#SBATCH --account=hhy@h100      # echo $IDRPROJ
#SBATCH -C h100                 # partition gpu_p6 : noeud quadri-GPU H100 80 Go
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
# Le nombre de CPU par tache doit etre adapte en fonction de la partition utilisee. Sachant
# qu'ici on ne reserve qu'un seul GPU (soit 1/4 des GPU du noeud), l'ideal est de reserver 1/4
# des CPU du noeud pour la seule tache :
#SBATCH --cpus-per-task=24           # nombre de CPU par tache pour gpu_p6 (1/4 des CPU du noeud 4-GPU H100)
#SBATCH --hint=nomultithread        # réservation des cœurs physiques (pas d'hyperthreading)
#SBATCH --time=05:00:00
#SBATCH --qos=qos_gpu_h100-t3    # qos_cpu-t3 (default, 20h), qos_cpu-t4 (100h), qos_cpu-dev (2h)


RUN_DIR="$SCRATCH/ADiT"
RUN_SCRIPT="./adit/train.py"
ADIT_SIZE="XS"
GNN_TYPE='gat'  # gcn / gat

EPOCHS=33

CKPT="outputs/pretrain_gat_phase2/gat${SLURM_ARRAY_TASK_ID}_denoising_${ADIT_SIZE}/checkpoints/last.ckpt"

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
module purge
module load arch/h100
source ~/.bashrc
conda deactivate
conda activate $WORK/envs/adit
# ---- Run it...
cd "$RUN_DIR"
set -x
echo 'Launching script'

srun python "$RUN_SCRIPT" \
    experiment=new_denoise_${ADIT_SIZE}_pdb_fixed_0_5 \
    +model/gnn=$GNN_TYPE \
    trainer.devices=$SLURM_NTASKS_PER_NODE \
    trainer.min_epochs=$EPOCHS \
    trainer.max_epochs=$EPOCHS \
    task_name=${GNN_TYPE}${SLURM_ARRAY_TASK_ID}_denoising_${ADIT_SIZE} \
    logger=many_loggers \
    ++logger.wandb.name=${GNN_TYPE}${SLURM_ARRAY_TASK_ID}_${ADIT_SIZE}\
    ++logger.wandb.id=${GNN_TYPE}${SLURM_ARRAY_TASK_ID}_pretrain_$SLURM_JOB_ID \
    ++logger.wandb.project="pretrain"
