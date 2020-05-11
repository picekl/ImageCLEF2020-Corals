#!/bin/sh

export CUDA_VISIBLE_DEVICES=1

CHECKPOINT_DIR=../checkpoints/800x800_inaturalist/
CONFIG_PATH=../configs/pipeline_faster_rcnn_resnet50_800x800_inaturalist.config
CHECKPOINT_STEPS=500

python ../object_detection/model_main.py \
    --model_dir "$CHECKPOINT_DIR" \
    --pipeline_config_path "$CONFIG_PATH" \
    --checkpoint_steps "$CHECKPOINT_STEPS"
