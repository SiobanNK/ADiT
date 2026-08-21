#!/bin/bash
#SBATCH --job-name=pretrain_gat
#SBATCH --output=./logs/%x%a_%j.log    # x: job name , j: job ID, a: array task ID
#SBATCH --error=./logs/%x%a_%j.err     # ${SLURM_ARRAY_TASK_ID}
#SBATCH --array=0,7,10,13,14,15
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

EPOCHS=33

CKPT="outputs/pretrain_gat_phase2/gat${SLURM_ARRAY_TASK_ID}_denoising_${ADIT_SIZE}/checkpoints/last.ckpt"

# Nombre à convertir (0-64), sur 8 bits ici -> adapte NBITS si besoin
NUM=$SLURM_ARRAY_TASK_ID
NBITS=8

# Conversion en binaire, complété par des zéros à gauche
BIN=$(printf '%0*d' "$NBITS" "$(echo "obase=2;$NUM" | bc)")

# Construction de la liste Hydra-compatible [False, True, ...]
LIST="["
for (( i=0; i<${#BIN}; i++ )); do
    bit=${BIN:$i:1}
    if [ "$bit" = "1" ]; then
        LIST+="True"
    else
        LIST+="False"
    fi
    if [ $i -lt $((${#BIN}-1)) ]; then
        LIST+=","
    fi
done
LIST+="]"

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
    ckpt_path=$CKPT \
    +model/gnn=gat \
    model.gnn_positions=$LIST \
    trainer.devices=$SLURM_NTASKS_PER_NODE \
    trainer.min_epochs=$EPOCHS \
    trainer.max_epochs=$EPOCHS \
    task_name=gat${SLURM_ARRAY_TASK_ID}_denoising_${ADIT_SIZE} \
    logger=many_loggers \
    ++logger.wandb.name=gat${SLURM_ARRAY_TASK_ID}_${ADIT_SIZE}_pretrain_nm\
    ++logger.wandb.id=gat${SLURM_ARRAY_TASK_ID}_pretrain_$SLURM_JOB_ID \
    ++logger.wandb.project="pretrain"

    # +model.net.atom_neighbour_radius="0.5" \
    # +model.net.token_neighbour_radius="1.0" \
    # +model.net.atom_gat_positions=$ATOM_PARAM    \
    # +model.net.token_gat_positions=$TOKEN_PARAM \
