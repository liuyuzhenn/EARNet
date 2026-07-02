#!/bin/bash

GPUS=8
PORT=${PORT:-29500}

torchrun --nproc_per_node=$GPUS --master_port=$PORT train.py ${@:2}