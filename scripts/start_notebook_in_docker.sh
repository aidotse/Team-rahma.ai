#!/bin/bash
nohup jupyter notebook --port=2020 --ip=0.0.0.0 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password='' &>/dev/null &

