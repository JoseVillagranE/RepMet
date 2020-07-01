#!/bin/bash

GPU=$1 
shift
DIR=/project 

echo GPU = $GPU

NV_GPU="$GPU" nvidia-docker run --rm --name repmet_training -v $HOME/:$DIR:rw -t repmet $@