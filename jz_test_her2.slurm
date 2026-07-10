#!/bin/bash
#SBATCH --job-name=her2
#SBATCH --output=./logs/%x_%j.log    # x: job name , j: job ID, a: array task ID
#SBATCH --error=./logs/%x_%j.err     # ${SLURM_ARRAY_TASK_ID}
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


# --qos=qos_cpu-dev     # qos_cpu-t3 (default, 20h), qos_cpu-t4 (100h), qos_cpu-dev (2h)


# Protein - Ligand : "lba"
# Drug - Target : "davis"
# Protein - Protein : "skempi"
# Antibody-Antigen : "her2" (test only)


RUN_DIR="$SCRATCH/ADiT"
RUN_SCRIPT="./adit/test.py"
DATA="her2"
FINETUNING_DATE="2026-07-07"
FINETUNING="skempi_S_split_0_10-34-52"
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
    data=$DATA \
    task_name=${DATA}_${ADIT_SIZE} \
    ++trainer.devices=1 \
    ++data.batch_size=1 \
    ++model.save_file=result_${DATA}_$ADIT_SIZE.pkl \
    ckpt_path=outputs/${FINETUNING_DATE}/${FINETUNING}/checkpoints/last.ckpt \
    logger=many_loggers \
    ++logger.wandb.name=${DATA}_${ADIT_SIZE} \
    ++logger.wandb.id=${DATA}_${ADIT_SIZE}_${SLURM_JOB_ID}
