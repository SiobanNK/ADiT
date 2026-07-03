#!/bin/bash
#SBATCH --job-name=esm
#SBATCH --output=./logs/%x_%j.log    # x: job name , j: job ID, a: array task ID
#SBATCH --error=./logs/%x_%j.err
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
#SBATCH --time=02:00:00
## SBATCH --qos=qos_cpu-dev     # qos_cpu-t3 (default, 20h), qos_cpu-t4 (100h), qos_cpu-dev (2h)



RUN_DIR="$SCRATCH/ADiT"
RUN_SCRIPT="./scripts/dump_esm_repr.py"

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
module load pytorch-gpu/py3/2.5.0
# VENV="$WORK/envs/adit"
# source $WORK/envs/adit/bin/activate
# echo "python: $(which python)"
# srun which python
# srun python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# echo "WORK var: $WORK"
# echo "VENV var: $VENV"
# ls -la "$VENV/bin/activate"

# ---- Run it...
cd "$RUN_DIR"
echo 'Launching script'

# srun bash -c "source $VENV/bin/activate && python $RUN_SCRIPT"
srun python "$RUN_SCRIPT" # useless?: --config-name=reference
