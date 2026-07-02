#!/bin/bash
#SBATCH --job-name=skempi_metric
#SBATCH --output=./logs/%x_%j.log    # x: job name , j: job ID, a: array task ID
#SBATCH --error=./logs/%x_%j.err
#SBATCH --mail-type=END                 # Send email at job completion
#SBATCH --mail-user=sioban.nieradzik-kozic@etu.minesparis.psl.eu
#SBATCH --account=hhy@h100   # echo $IDRPROJ

#SBATCH -C h100
#SBATCH --partition=gpu_p6    # prepost, visu, compil, archive
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --hint=nomultithread        # réservation des cœurs physiques (pas d'hyperthreading)
## SBATCH --qos=qos_cpu-dev     # qos_cpu-t3 (default, 20h), qos_cpu-t4 (100h), qos_cpu-dev (2h)



RUN_DIR="$SCRATCH/ADiT"
RUN_SCRIPT="./adit/train.py"
SPLIT=2
DATA="skempi"
ADIT_SIZE="S"

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
module purge
module load python/3.10.4 cuda/12.4.0 cudnn/8.9.7.29-cuda
source $SCRATCH/envs/adit/bin/activate

# ---- Run it...
cd "$RUN_DIR"
echo 'Launching script'

srun python "$RUN_SCRIPT"
    experiment=${DATA}_$ADIT_SIZE \
    ++data.dataset.test_split=split_$SPLIT \
    task_name=${DATA}_${ADIT_SIZE}_split_$SPLIT \
    ckpt_path=ckpts/adit_$ADIT_SIZE.ckpt
