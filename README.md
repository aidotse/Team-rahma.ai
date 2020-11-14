# [Adipocyte Cell Imaging Challenge](https://www.ai.se/en/challenge)

AstraZeneca and AI Sweden are challenging the AI community to solve the problem of labeling cell images without requiring toxic preprocessing of cell cultures. 

**Rähmä.ai solution**

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

**In the repository root** (or in Docker container `/main/`), run

```sh
MODEL_DIR=$(pwd)/models/20201109-193651_resnet50
INPUT_DIR=$(pwd)/tmp/test_slides/20x_input
OUTPUT_DIR=$(pwd)/tmp/test_output/
ZOOM_LEVEL=20

python ./src/inference.py -model_dir $MODEL_DIR -zoom $ZOOM_LEVEL -predict_dir $INPUT_DIR -output_dir $OUTPUT_DIR
```

or 

**As a docker script (preferrable)**

```sh
CODE_DIR=$(pwd) #<-default
MODEL_DIR=$(pwd)/tmp/20201109-193651_resnet50 #<-default
INPUT_DIR=$(pwd)/tmp/test_slides/20x_input #<-change this
OUTPUT_DIR=$(pwd)/tmp/test_output/ #<-change this
ZOOM_LEVEL=20 #<-change this
DOCKER_IMAGE=raehmae_docker_image:latest #<-default

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

**Notice:** The inference script will print the inference time spend working the inference on images in `input_dir` that includes:

+ splitting the brightfield images into tiles

+ inference with the ensemble

+ stitching the output tiles into fluerecence images

but exludes the time spend in:

- loading ensemble weights into memory.

The inference print looks like this:

```sh
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
MODEL_DIR=$(pwd)/tmp/20201109-193651_resnet50
OUTPUT_DIR=$(pwd)/tmp/20201109-193651_resnet50

python ./src/evaluate.py \
    --model_dir=$MODEL_DIR \
    --output_dir=$OUTPUT_DIR
    
```

or 

**As a docker script (preferrable)**
```sh
CODE_DIR=$(pwd)
MODEL_DIR=$(pwd)/models/20201109-193651_resnet50
OUTPUT_DIR=$(pwd)/models/20201109-193651_resnet50
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
