#!/bin/bash
#SBATCH --pty                   # obtenir un terminal interactif et la commande passée à srun est un shell (ici bash).
#SBATCH --nodes=1               # réservation des cœurs physiques (pas d'hyperthreading)
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4      # La mémoire allouée pour le job est proportionnelle au nombre de cœurs/CPU demandés.
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00         # temps exécution maximum
#SBATCH --hint=nomultithread

srun bash

# Une fois le terminal opérationnel, vous pouvez lancer vos exécutables de la manière habituelle ./votre_executable.
# Pour quitter le mode interactif, utilisez la commande exit.
