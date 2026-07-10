#!/bin/bash
#SBATCH --job-name=test_protein_protein
#SBATCH --output=./logs/%x_%j.log    # x: job name , j: job ID, a: array task ID
#SBATCH --error=./logs/%x_%j.err
#SBATCH --mem=15G
#SBATCH --time=00:10:00
#SBATCH --partition=cbio-gpu
#SBATCH --gres=gpu:P100:1               # A40 (48Go) , P100 (16Go, fastest), T4 (16Go, especially designed for inference), K80
#SBATCH --nodes=1
##SBATCH --cpus-per-task=1
#SBATCH --mail-type=END                 # Send email at job completion
#SBATCH --mail-user=sioban.nieradzik-kozic@etu.minesparis.psl.eu

RUN_DIR="/cluster/CBIO/home/snieradzikkozic/ADiT"
RUN_SCRIPT="./adit/test.py"
SPLIT=1            # ADAPT
ADIT_SIZE="S"       # ADAPT
DATA="skempi"       # ADAPT

FINETUNING_DATE="2026-06-19"    # ADAPT
FINETUNING_HOUR="20-49-11"           # ADAPT
PATH_TO_FINETUNED_CKPT="outputs/${FINETUNING_DATE}/${DATA}_${ADIT_SIZE}_split_${SPLIT}_${FINETUNING_HOUR}/checkpoints/last.ckpt"

# Antibody - Antigen : "her2"

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
# ---- Run it...
source /cluster/CBIO/home/snieradzikkozic/miniconda3/etc/profile.d/conda.sh
conda activate adit
cd "$RUN_DIR"
echo 'Launching script'

export NUMEXPR_MAX_THREADS=8

srun python "$RUN_SCRIPT" \
    experiment=${DATA}_$ADIT_SIZE \
    task_name=${DATA}_${ADIT_SIZE} \
    ++trainer.devices=1 \
    ++data.batch_size=1 \
    ++data.dataset.test_split=split_$SPLIT \
    ++model.save_file=outputs/result_split_$SPLIT.pkl \
    ckpt_path=$PATH_TO_FINETUNED_CKPT


# bash test.sh experiment=skempi_L ++trainer.devices=1 ++data.batch_size=1 ++data.dataset.test_split=split_0 ++model.save_file=result_split_0.pkl ckpt_path="PATH_TO_FINETUNED_CKPT"
