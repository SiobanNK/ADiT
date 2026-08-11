#!/bin/bash
#SBATCH --job-name=pretrain_rbf
#SBATCH --output=./logs/%x_%j.log    # x: job name , j: job ID, a: array task ID
#SBATCH --error=./logs/%x_%j.err
#SBATCH --mem=45G
#SBATCH --time=50:00:00
#SBATCH --partition=cbio-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:A40:2               # A40 (48Go) , P100 (16Go, fastest), T4 (16Go, especially designed for inference), K80
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=END                 # Send email at job completion
#SBATCH --mail-user=sioban.nieradzik-kozic@etu.minesparis.psl.eu




RUN_DIR="/cluster/CBIO/home/snieradzikkozic/ADiT"
RUN_SCRIPT="./adit/train.py"
ADIT_SIZE="S"

COORD_ENCODING="rbf"     # onehot , rbf , null
D_MAX=22
EPOCHS=33

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
    ++model.net.token_coord_encoder=$COORD_ENCODING \
    ++model.net.d_max=$D_MAX \
    ++trainer.devices=$SLURM_NTASKS_PER_NODE \
    ++trainer.min_epochs=$EPOCHS \
    ++trainer.max_epochs=$EPOCHS \
    ++task_name=denoising_${ADIT_SIZE}_${COORD_ENCODING}_d$D_MAX \
    logger=many_loggers \
    ++logger.wandb.name=pretrain_${COORD_ENCODING}_d$D_MAX \
    ++logger.wandb.id=pretrain_${COORD_ENCODING}_d$D_MAX_$SLURM_JOB_ID \
    ++logger.wandb.project="pretrain"
