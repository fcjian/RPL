#!/bin/bash

cd ..

# custom config
TRAINER=RPL

DATASET=$1
CFG=$2  # config file
DATA=$3
NCTX=$4  # number of context tokens
SHOTS=$5 

for SEED in 3
do
    DIR=output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}
    if [ -d "$DIR" ]; then
        echo "Oops! The results exist at ${DIR} (so skip this job)"
    else
        python -m pdb train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        TRAINER.RPL.N_CTX ${NCTX} \
        DATASET.NUM_SHOTS ${SHOTS}
    fi
done
