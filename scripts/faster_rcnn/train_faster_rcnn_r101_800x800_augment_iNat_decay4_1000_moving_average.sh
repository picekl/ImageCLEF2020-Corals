#!/bin/sh

export CUDA_VISIBLE_DEVICES=7

CHECKPOINT_DIR=../checkpoints/faster_rcnn_resnet101_800x800_augment_iNat_decay4_1000_moving_average.config
CONFIG_PATH=../configs/pipeline_faster_rcnn_resnet101_800x800_augment_iNat_decay4_1000_moving_average.config
CHECKPOINT_STEPS=500

python ../object_detection/model_main.py \
    --model_dir "$CHECKPOINT_DIR" \
    --pipeline_config_path "$CONFIG_PATH" \
    --checkpoint_steps "$CHECKPOINT_STEPS"
