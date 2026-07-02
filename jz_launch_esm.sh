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
##SBATCH --cpus-per-task=1
#SBATCH --mem=15G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:H100:1
## SBATCH --qos=qos_cpu-dev     # qos_cpu-t3 (default, 20h), qos_cpu-t4 (100h), qos_cpu-dev (2h)



RUN_DIR="$WORK/ADiT"
RUN_SCRIPT="./scripts/skempi_metric.py"
INPUT_DIR="./outputs"  # contain skempi_result_0.pkl
# ---- Welcome...
echo '-------------------------------------'
echo "Start : $0"
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
module load python/3.10.4 cuda/12.4.0 cudnn/8.9.7.29-cuda
source $WORK/envs/adit/bin/activate
cd "$RUN_DIR"
echo 'Launching script'
srun python "$RUN_SCRIPT" --input_dir $INPUT_DIR
