#!/bin/bash
#SBATCH --job-name=batch_example                                                        # Job name
#SBATCH --output=/mnt/data4/mnajm/CFTR_PROJECT/log/group_inter_stats_%j.log             # Output log
#SBATCH --error=/mnt/data4/mnajm/CFTR_PROJECT/log/group_inter_stats_%j.err              # Error log
#SBATCH --mem 20000                                                                     # Job memory request
#SBATCH --gres=gpu:A40:1                # Number of GPUs + type of GPUs that you want
#SBATCH -p cbio-gpu                     # Name of the partition to use
#SBATCH --nodelist=node006              # Nodes that you want (better to use gres instead)
#SBATCH --exclude=node009               # Alternatively, nodes that you do not want
#SBATCH --cpus-per-task=4               # CPU cores per process (default 1, typically 4 or 5 - do not use more to let space for other people)

#MODULE_ENV="pytorch-gpu/py3/2.8.0"
RUN_DIR="$WORK/rna/rna_pretraining/"
RUN_SCRIPT="./scripts/dump_esm_repr.py"
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
#echo "With env. : $MODULE_ENV"
echo '--------- --------------------------'
# ---- Module
export CUDA_VISIBLE_DEVICES=0
module purge
module load "$MODULE_ENV"
# ---- Run it...
cd "$RUN_DIR"
echo 'Launching script'
srun python "$RUN_SCRIPT" --config-name=reference # dump_esm_repr.py drugbank_v5.1.5 S0h --non_balanced --job_id $SLURM_JOB_ID

# Write in the cluster, in the same folder as the py file : sbatch batch_example.sh
