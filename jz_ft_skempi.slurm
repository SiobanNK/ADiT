#!/bin/bash
#SBATCH --job-name=skempi               # lba, skempi, davis, her2
#SBATCH --output=./logs/%x_%j_%a.log    # x: job name , j: job ID, a: array task ID
#SBATCH --error=./logs/%x_%j_%a.err     # ${SLURM_ARRAY_TASK_ID}
#SBATCH --array=0-2
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
#SBATCH --time=20:00:00
## SBATCH --qos=qos_cpu-dev     # qos_cpu-t3 (default, 20h), qos_cpu-t4 (100h), qos_cpu-dev (2h)


# Protein - Ligand : "lba"
# Drug - Target : "davis"
# Protein - Protein : "skempi"
# Antibody-Antigen : "her2"


RUN_DIR="$SCRATCH/ADiT"
RUN_SCRIPT="./adit/train.py"
SPLIT="$SLURM_ARRAY_TASK_ID"
DATA="$SLURM_JOB_NAME"
ADIT_SIZE="S"

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
# module load pytorch-gpu/py3/2.5.0
# source $SCRATCH/envs/adit/bin/activate
source ~/.bashrc
conda deactivate
conda activate $WORK/envs/adit
# ---- Run it...
cd "$RUN_DIR"
set -x
echo 'Launching script'

srun python "$RUN_SCRIPT" \
    experiment=${DATA}_$ADIT_SIZE \
    data.dataset.test_split=split_$SPLIT \
    ++task_name=${DATA}_${ADIT_SIZE}_split_$SPLIT \
    ckpt_path=ckpts/adit_$ADIT_SIZE.ckpt \
    logger=many_loggers \
    ++logger.wandb.name=${DATA}_$ADIT_SIZE_$SPLIT \
    ++logger.wandb.id=${DATA}_$ADIT_SIZE_${SLURM_JOB_ID}_$SLURM_ARRAY_TASK_ID
