#!/bin/bash
python ./src/calculate_statistics_preview.py \
    --input_images_dir=/data/ \
    --config=./configs/train_config.json

python ./src/create_tile_dataset_preview.py \
    --input_images_dir=/data/ \
    --config=./configs/train_config.json