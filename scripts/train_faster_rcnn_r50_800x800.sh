#!/bin/sh

export CUDA_VISIBLE_DEVICES=0

CHECKPOINT_DIR=~/coral/model/out_faster_rcnn_r50_800x800
CONFIG_PATH=../configs/pipeline_faster_rcnn_resnet50_800x800.config
CHECKPOINT_STEPS=99

python ../object_detection/model_main.py \
    --model_dir "$CHECKPOINT_DIR" \
    --pipeline_config_path "$CONFIG_PATH" \
    --checkpoint_steps "$CHECKPOINT_STEPS"
