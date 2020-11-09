#!/bin/bash
python ./src/calculate_statistics.py \
    --input_images_dir=/data/ \
    --config=./configs/train_config.json

python ./src/create_tile_dataset.py \
    --input_images_dir=/data/ \
    --config=./configs/train_config.json