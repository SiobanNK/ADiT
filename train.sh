#!/bin/bash
WORKDIR=$(pwd)

dt=$(date '+%d/%m/%Y-%H:%M:%S')
echo "[$0] >>> Starttime => ${dt}"

#########################
####### Routine #########
#########################
export NUMEXPR_MAX_THREADS=8


echo ">>> Exec: python adit/train.py $@"
python adit/train.py $@ 
