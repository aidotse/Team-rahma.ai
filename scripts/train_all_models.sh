#!/bin/bash
python ./src/train.py \
    --config=./configs/train_config.json \
    --bs=12 \
    --val_folds=0,2 \
    --head_epochs=1 \
    --head_lr=0.001 \
    --unfreezed_mse_epochs=15 \
    --unfreezed_mse_lr=0.0001 \
    --unfreezed_combination_loss_epochs=10 \
    --unfreezed_combination_loss_lr=0.0001 \
    --base_arch=resnet50 \ &&

