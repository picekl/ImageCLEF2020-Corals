#!/bin/sh

export CUDA_VISIBLE_DEVICES=0

CHECKPOINT_DIR=../checkpoints/pipeline_faster_rcnn_resnet101_800x800_inat_augment
CONFIG_PATH=../configs/pipeline_faster_rcnn_resnet101_800x800_inat_augment.config
CHECKPOINT_STEPS=500

python ../object_detection/model_main.py \
    --model_dir "$CHECKPOINT_DIR" \
    --pipeline_config_path "$CONFIG_PATH" \
    --checkpoint_steps "$CHECKPOINT_STEPS"
