#!/bin/bash

CUDA_DEVICE=0,1,2,3
DATADIR="./data"
LOGDIR="./logs/imagenet100/simclr"

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python pretrain.py \
    --logdir $LOGDIR \
    --framework simclr \
    --dataset imagenet100 \
    --datadir $DATADIR \
    --model resnet50 \
    --batch-size 256 \
    --max-epochs 500 \
    --base-lr 0.03 --wd 5e-4 \
    --expert 8 \
    --inv_T 0.2 \
    --eq_T 0.2 \
    --ckpt-freq 100 --eval-freq 100 \
    --num-workers 4 --distributed