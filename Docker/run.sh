#!/bin/bash
# File              : run.sh
# Author            : Sheetal Reddy <sheetal.reddy@ai.se>
# Date              : 23.10.2020
# Last Modified Date: 02.11.2020
# Last Modified By  : Joni Juvonen <joni.juvonen@silo.ai>

DATA_DIR=$(pwd)/../../astra_data_readonly
CODE_DIR=$(pwd)/..

nvidia-docker  run   \
	-v $DATA_DIR:/data \
	-v $CODE_DIR:/main \
        -p 2020:2020 \
        -d \
	-it raehmae_docker_image \
	bash 



