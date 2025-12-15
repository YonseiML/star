#!/bin/bash

CUDA_DEVICE=0
CKPT="./logs/stl10/simclr/ckpt-200.pth"
DATADIR="./data"

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python linear_eval.py \
    --pretrain-data stl10 \
    --ckpt $CKPT \
    --model resnet18 \
    --dataset cifar10 \
    --datadir $DATADIR \
    --metric top1