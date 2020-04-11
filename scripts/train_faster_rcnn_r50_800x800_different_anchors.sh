#!/bin/sh

export CUDA_VISIBLE_DEVICES=3

CHECKPOINT_DIR=../checkpoints/faster_rcnn_r50_800x800_different_anchors
CONFIG_PATH=../configs/pipeline_faster_rcnn_resnet50_800x800_different_anchors.config
CHECKPOINT_STEPS=500

python ../object_detection/model_main.py \
    --model_dir "$CHECKPOINT_DIR" \
    --pipeline_config_path "$CONFIG_PATH" \
    --checkpoint_steps "$CHECKPOINT_STEPS"
