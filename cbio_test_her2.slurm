#!/bin/bash
#SBATCH --job-name=her2
#SBATCH --output=./logs/%x_%j.log    # x: job name , j: job ID, a: array task ID
#SBATCH --error=./logs/%x_%j.err
#SBATCH --mem=45G
#SBATCH --time=06:00:00
#SBATCH --partition=cbio-gpu
#SBATCH --gres=gpu:P100:1               # A40 (48Go) , P100 (16Go, fastest), T4 (16Go, especially designed for inference), K80
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4               # default 1, typically 4 or 5 - do not use more
#SBATCH --mail-type=END                 # Send email at job completion
#SBATCH --mail-user=sioban.nieradzik-kozic@etu.minesparis.psl.eu


# --qos=qos_cpu-dev     # qos_cpu-t3 (default, 20h), qos_cpu-t4 (100h), qos_cpu-dev (2h)


# Protein - Ligand : "lba"
# Drug - Target : "davis"
# Protein - Protein : "skempi"
# Antibody-Antigen : "her2" (test only)


RUN_DIR="/cluster/CBIO/home/snieradzikkozic/ADiT"
RUN_SCRIPT="./adit/test.py"
DATA="her2"
FINETUNING_DATE="2026-06-19"
FINETUNING="skempi_S_split_0_11-49-13"
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
source ~/.bashrc
conda activate adit
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
