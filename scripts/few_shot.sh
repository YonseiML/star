#!/bin/bash

CUDA_DEVICE=0
CKPT="./logs/stl10/simclr/ckpt-200.pth"
DATADIR="./data"

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python few_shot.py \
    --pretrain-data stl10 \
    --ckpt $CKPT \
    --model resnet18 \
    --dataset fc100 \
    --datadir $DATADIR \
    # --K 5