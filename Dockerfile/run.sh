#!/bin/bash

GPU=$1 
shift
DIR=/project 

NV_GPU="$GPU" nvidia-docker run --rm --name repmet_training -v `pwd`/:$DIR:rw -t repmet $@