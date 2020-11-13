# [Adipocyte Cell Imaging Challenge](https://www.ai.se/en/challenge)

AstraZeneca and AI Sweden are challenging the AI community to solve the problem of labeling cell images without requiring toxic preprocessing of cell cultures. 

**Rähmä.ai solution**

--------------------------------------

## Installation

**BUILD DOCKER**

```bash
cd Docker && sh build.sh
```

## Training TODO!

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

## Train




## Inference

1. In the repository root (or in Docker container `/main/`), run
```sh
python ./src/inference.py -model_dir {MODEL_DIR} -zoom {ZOOM_LEVEL} -predict_dir {PREDICT_DIR} -output_dir {OUTPUT_DIR}
```

or 

2. As a docker script
```sh
CODE_DIR=$(pwd)
MODEL_DIR=$(pwd)/tmp/20201112-225630_resnet34
INPUT_DIR=$(pwd)/tmp/test_slides/20x_input
OUTPUT_DIR=$(pwd)/tmp/test_output/
ZOOM=20
DOCKER_IMAGE=raehmae_docker_image:latest

docker run \
  -v $CODE_DIR:/main/:ro \
  -v $MODEL_DIR:/model_dir/:ro \
  -v $INPUT_DIR:/input_dir/:ro \
  -v $OUTPUT_DIR:/output_dir/ \
  $DOCKER_IMAGE \
  python3 src/inference.py \
    --model_dir=/model_dir \
    --zoom=$ZOOM \
    --predict_dir=/input_dir \
    --output_dir=/output_dir
```

where

* `MODEL_DIR`: (Relative) path to models, default `'../models/20201109-193651_resnet50'`.

* `ZOOM_LEVEL`: Magnification integer, one of 20, 40, 60

* `PREDICT_DIR`: (Relative) path to *C04.tif files to predict. All slides can be dumped into the same input folder. For each unique slide id in the dir there must be 7 tiffs associated to them.

* `OUTPUT_DIR`: Directory where predicted fluorescence tiffs are saved to. All tiffs will be dumped into the same output folder.

* `DOCKER_IMAGE`: Name of the previously build docker image.


## Evaluate

Generate the evaluation test set by running

```sh
python ./src/create_test_set.py -input_images_dir {PATH/TO/DATA}
```

After this, the models can evaluated by

1. In the repository root (or in Docker container `/main/`), run

```sh
python ./src/evaluate.py \
    -model_dir {MODEL_DIR} \
    -predict_dir {PREDICT_DIR} \
    -output_dir {OUTPUT_DIR}
```

or 

2. As a docker script
```sh
docker exec {DOCKER_IMAGE} python3 evaluate.py \
    -model_dir {MODEL_DIR} \
    -output_dir {OUTPUT_DIR}
```

where

* `MODEL_DIR`: (Relative) path to models

* `OUTPUT_DIR`: Directory where the evaluation results will be saved

* `DOCKER_IMAGE`: Name of the previously build docker image.
