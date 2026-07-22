#!/bin/bash
#SBATCH --job-name=lba_60
#SBATCH --output=./logs/%x_%j_%a.log    # x: job name , j: job ID, a: array task ID
#SBATCH --error=./logs/%x_%j_%a.err
#SBATCH --array=0-10
#SBATCH --mem=45G
#SBATCH --time=3:00:00
#SBATCH --partition=cbio-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:A40:2               # A40 (48Go) , P100 (16Go, fastest), T4 (16Go, especially designed for inference), K80
##SBATCH --cpus-per-task=1
#SBATCH --mail-type=END                 # Send email at job completion
#SBATCH --mail-user=sioban.nieradzik-kozic@etu.minesparis.psl.eu


# Protein - Ligand : "lba"
# Drug - Target : "davis"
# Protein - Protein : "skempi"
# Antibody-Antigen : "her2"


RUN_DIR="/cluster/CBIO/home/snieradzikkozic/ADiT"
RUN_SCRIPT="./adit/train.py"
DATA="lba"
ADIT_SIZE="S"
IDENTITY=60 # 30, 60
EPOCHS=100
DROPOUT=0.5

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
echo 'Launching script'


srun python "$RUN_SCRIPT" \
    experiment=${DATA}_$ADIT_SIZE \
    task_name=${DATA}_${ADIT_SIZE} \
    ++trainer.devices=$SLURM_NTASKS_PER_NODE \
    ++trainer.max_epochs=$EPOCHS \
    ++data.batch_size=16 \
    ++data.num_workers=$SLURM_CPUS_PER_TASK \
    ++data.dataset.split_identity_threshold=identity_$IDENTITY \
    ++model.net.dropout=$DROPOUT \
    ckpt_path=ckpts/adit_$ADIT_SIZE.ckpt \
    logger=many_loggers \
    ++logger.wandb.name=${DATA}${IDENTITY}_$ADIT_SIZE \
    ++logger.wandb.id=${DATA}${IDENTITY}_${SLURM_JOB_ID}_$SLURM_ARRAY_TASK_ID \
    ++logger.wandb.project=${DATA} \
    ++logger.wandb.offline=False \
    # ++trainer.reload_dataloaders_every_n_epochs=1 \
    # ++trainer.use_distributed_sampler=False \
    # ++data.max_pair_budget=250000 \
    # ++data.max_total_atoms=15000 \

# trainer.devices : S : 2, M : 4, L : 4
