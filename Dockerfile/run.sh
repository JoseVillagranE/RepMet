#!/bin/bash

GPU=$1 
shift
DIR=/Repmet/models 

echo GPU = $GPU
echo Container Directory: $DIR

NV_GPU="$GPU" nvidia-docker run --rm --name repmet_training -v $HOME/:$DIR -t repmet $@
