#!/bin/sh

export CUDA_VISIBLE_DEVICES=5

CHECKPOINT_DIR=../checkpoints/faster_ssd_resnet50_v1_fpn_800x800
CONFIG_PATH=../configs/pipeline_ssd_resnet50_v1_fpn.config
CHECKPOINT_STEPS=500

python ../object_detection/model_main.py \
    --model_dir "$CHECKPOINT_DIR" \
    --pipeline_config_path "$CONFIG_PATH" \
    --checkpoint_steps "$CHECKPOINT_STEPS"
