#!/bin/bash

CUDA_DEVICE=0
DATADIR="./data"
LOGDIR="./logs/stl10/simclr"

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python pretrain.py \
    --logdir $LOGDIR \
    --framework simclr \
    --dataset stl10 \
    --datadir $DATADIR \
    --model resnet18 \
    --batch-size 256 \
    --max-epochs 200 \
    --base-lr 0.03 --wd 5e-4 \
    --expert 16 \
    --inv_T 0.2 \
    --eq_T 0.2 \
    --num-workers 16