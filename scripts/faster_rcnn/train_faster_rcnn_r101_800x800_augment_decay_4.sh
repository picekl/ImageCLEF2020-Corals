#!/bin/sh

export CUDA_VISIBLE_DEVICES=7

CHECKPOINT_DIR=../../checkpoints/faster_rcnn_resnet101_800x800_augment_decay4_AG
CONFIG_PATH=../../configs/faster-rcnn/pipeline_faster_rcnn_resnet101_800x800_augment_decay4.config
CHECKPOINT_STEPS=1000

python ../../object_detection/model_main.py \
    --model_dir "$CHECKPOINT_DIR" \
    --pipeline_config_path "$CONFIG_PATH" \
    --checkpoint_steps "$CHECKPOINT_STEPS"
