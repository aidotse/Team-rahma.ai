# [Adipocyte Cell Imaging Challenge](https://www.ai.se/en/challenge)

AstraZeneca and AI Sweden are challenging the AI community to solve the problem of labeling cell images without requiring toxic preprocessing of cell cultures. 

**Rähmä.ai solution**

--------------------------------------

## Installation

1. Create conda python environment for managing cuda version.
```bash
conda create --name cuda102 python=3.7
conda activate cuda102
conda install cudatoolkit=10.2  && conda install cudnn=7.6
```

2. Install python virtualenv and pip requirements
```bash
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
```

## Get dataset statistics

This will calculate and save the dataset statistics to `./configs/data_statistics.json`.
Other methods may rely on this output file.

```bash
python ./src/calculate_statistics.py \
    --input_images_dir=./input/images_for_preview/ \
    --output_file=./configs/data_statistics.json
```

## Create tile dataset

```bash
python ./src/create_tile_dataset.py
```