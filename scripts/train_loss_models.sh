#!/bin/bash
python ./src/train_loss_models.py \
    --config=./configs/train_config.json \
    --bs=4 \
    --ep_head=5 \
    --lr_head=0.001 \
    --ep_unfreezed=25 \
    --lr_unfreezed=0.0001 \