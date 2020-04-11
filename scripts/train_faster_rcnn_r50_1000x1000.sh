#!/bin/sh

export CUDA_VISIBLE_DEVICES=1

CHECKPOINT_DIR=../checkpoints/faster_rcnn_r50_1000x1000
CONFIG_PATH=../configs/pipeline_faster_rcnn_resnet50_1000x1000.config
CHECKPOINT_STEPS=99

python3 ../object_detection/model_main.py \
    --model_dir "$CHECKPOINT_DIR" \
    --pipeline_config_path "$CONFIG_PATH" \
    --checkpoint_steps "$CHECKPOINT_STEPS"
