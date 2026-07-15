#!/bin/bash
#SBATCH --job-name=skempi_onehot
#SBATCH --output=./logs/%x_%j.log    # x: job name , j: job ID, a: array task ID
#SBATCH --error=./logs/%x_%j.err
#SBATCH --array=0-2
#SBATCH --mem=45G
#SBATCH --time=20:00:00
#SBATCH --partition=cbio-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A40:1               # A40 (48Go) , P100 (16Go, fastest), T4 (16Go, especially designed for inference), K80
##SBATCH --cpus-per-task=1
#SBATCH --mail-type=END                 # Send email at job completion
#SBATCH --mail-user=sioban.nieradzik-kozic@etu.minesparis.psl.eu




RUN_DIR="/cluster/CBIO/home/snieradzikkozic/ADiT"
RUN_SCRIPT="./adit/train.py"
DATA="lba"

COORD_ENCODING="onehot"     # onehot , rbf , null

# Protein - Ligand : "lba"
# Drug - Target : "davis"
# Protein - Protein : "skempi"

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
    experiment=skempi_S_coord_encoder \
    data.dataset.test_split=split_$SPLIT \
    ++task_name=pretrain_$COORD_ENCODING \
    ++model.net.token_coord_encoder=$COORD_ENCODING \
    logger=many_loggers \
    ++logger.wandb.name=skempi_${COORD_ENCODING} \
    ++logger.wandb.id=skempi_${COORD_ENCODING}_${SLURM_JOB_ID}
