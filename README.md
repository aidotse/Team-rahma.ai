# [Adipocyte Cell Imaging Challenge](https://www.ai.se/en/challenge)

AstraZeneca and AI Sweden are challenging the AI community to solve the problem of labeling cell images without requiring toxic preprocessing of cell cultures. 

**Team rähmä.ai solution**

--------------------------------------

## Installation

1. Build Docker from repository root

```bash
cd Docker && sh build.sh
```

2. Run Docker from repository root (optional, see details below)
```bash
cd Docker && sh run.sh
```


## Training

1. Create tile image dataset

```bash
DATA_DIR=$(pwd)/../../astra_data_readonly
CODE_DIR=$(pwd)/..
DOCKER_IMAGE=raehmae_docker_image:latest

nvidia-docker run \
    -v $DATA_DIR:/data \
    -v $CODE_DIR:/main \
    $DOCKER_IMAGE \
    scripts/create_tile_dataset.sh
```

2. Train the perceptual loss models with target images (siamese network)

```bash
CODE_DIR=$(pwd)/..
DOCKER_IMAGE=raehmae_docker_image:latest

nvidia-docker run \
    -v $CODE_DIR:/main \
    $DOCKER_IMAGE \
    scripts/train_loss_models.sh
```

3. Train the models: they will appear inside code dir in `models/{date}_{base_arch}`

```bash
CODE_DIR=$(pwd)/..
DOCKER_IMAGE=raehmae_docker_image:latest

nvidia-docker run \
    -v $CODE_DIR:/main \
    $DOCKER_IMAGE \
    scripts/train_all_models.sh
```


## Inference

First download `final_ensemble.zip` and unzip it inside `models` directory in the codebase. The contents inside `models` directory are:

```
models
|__ final_ensemble
    |__ _resnet50_fold_0
    |   |__ zoom_20
    |   |   |__ model.pth
    |   |   |__ model_config.json
    |   |   |__ ...
    |   |__ zoom_40
    |   |   |__ ...
    |   |__ zoom_60
    |   |   |__ ...
    |__ _resnet50_fold_2
    |   |__ ...
    |__ _resnet50_fold_3
        |__ ...
```

**In the repository root** (or in Docker container `/main/`), run

```sh
MODEL_DIR=$(pwd)/models/final_ensemble
INPUT_DIR=$(pwd)/tmp/test_slides/20x_input
OUTPUT_DIR=$(pwd)/tmp/test_output/
ZOOM_LEVEL=20

python ./src/inference.py -model_dir $MODEL_DIR -zoom $ZOOM_LEVEL -predict_dir $INPUT_DIR -output_dir $OUTPUT_DIR
```

or 

**As a docker script (preferrable)**

```sh
CODE_DIR=$(pwd) 
MODEL_DIR=$(pwd)/models/final_ensemble
INPUT_DIR=$(pwd)/tmp/test_slides/20x_input
OUTPUT_DIR=$(pwd)/tmp/test_output/
ZOOM_LEVEL=20
DOCKER_IMAGE=raehmae_docker_image:latest

docker run \
  -v $CODE_DIR:/main/:ro \
  -v $MODEL_DIR:/model_dir/:ro \
  -v $INPUT_DIR:/input_dir/:ro \
  -v $OUTPUT_DIR:/output_dir/ \
  $DOCKER_IMAGE \
  python3 src/inference.py \
    --model_dir=/model_dir \
    --zoom=$ZOOM_LEVEL \
    --predict_dir=/input_dir \
    --output_dir=/output_dir

```

---

Inference parameters explained:

*`CODE_DIR:` path to the codebase root

*`MODEL_DIR:` path to the checkpoints to use

*`INPUT_DIR:` path to the directory with all the input `*.tif` files to be inferred inside

*`OUTPUT_DIR:` path to the directory to generate the `*.tif` output files into

*`ZOOM_LEVEL:` one of 20, 40 or 60

*`DOCKER_IMAGE:` docker image name

---

**Notice:** The inference script will print the inference time spend working the inference on images in `input_dir` that includes:

+ splitting the brightfield images into tiles

+ inference with the ensemble

+ stitching the output tiles into fluerecence images

but exludes the time spend in:

- loading ensemble weights into memory.

The inference print looks like this:

```
~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 
Number of slides processed 10
Inference finished in 23.512750148773193 seconds
Average inference time for one slide 2.3512750148773193 seconds)
~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
```

## Evaluate

Generate the evaluation test set by running

```sh
python ./src/create_test_set.py -input_images_dir {DATA_ROOT}
```

After this, the models can evaluated by

**In the repository root** (or in Docker container `/main/`), run
```sh
MODEL_DIR=$(pwd)/models/final_ensemble
OUTPUT_DIR=$(pwd)/models/final_ensemble

python ./src/evaluate.py \
    --model_dir=$MODEL_DIR \
    --output_dir=$OUTPUT_DIR
    
```

or 

**As a docker script (preferrable)**
```sh
CODE_DIR=$(pwd)
MODEL_DIR=$(pwd)/models/final_ensemble
OUTPUT_DIR=$(pwd)/models/final_ensemble
DOCKER_IMAGE=raehmae_docker_image:latest

docker run \
  -v $CODE_DIR:/main/:ro \
  -v $MODEL_DIR:/model_dir/:ro \
  -v $OUTPUT_DIR:/output_dir/ \
  $DOCKER_IMAGE \
  python3 src/evaluate.py \
    --model_dir=/model_dir \
    --output_dir=/output_dir
```

---

**Important notice:** The internal evaluation uses [`CellProfiler`](https://github.com/CellProfiler/CellProfiler). Instead of simply cloning CP from the official repository, we had to fix some issues with the official code, and therefore needed to add CP into the codebase, in folder `CellProfiler/`. CP has their own non-MIT license, however, CP is only needed for the evaluation and the folder `CellProfiler/` can be simply deleted from the codebase. 

---

