#!/bin/sh

export CUDA_VISIBLE_DEVICES=1

CHECKPOINT_DIR=~/coral/model/out_faster_rcnn_r50_400x400
CONFIG_PATH=../configs/pipeline_faster_rcnn_resnet50_400x400.config
CHECKPOINT_STEPS=99

python ../object_detection/model_main.py \
    --model_dir "$CHECKPOINT_DIR" \
    --pipeline_config_path "$CONFIG_PATH" \
    --checkpoint_steps "$CHECKPOINT_STEPS"
