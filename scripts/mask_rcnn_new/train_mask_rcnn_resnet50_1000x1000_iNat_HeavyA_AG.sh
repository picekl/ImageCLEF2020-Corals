#!/bin/sh

export CUDA_VISIBLE_DEVICES=0

CHECKPOINT_DIR=../../checkpoints/mask_rcnn_resnet50_1000x1000_iNat_HeavyA_AG
CONFIG_PATH=../../configs/mask-rcnn-new/mask_rcnn_resnet50_1000x1000_iNat_HeavyA_AG.config
CHECKPOINT_STEPS=1950 #5 epochs

python ../../object_detection/model_main.py \
    --model_dir "$CHECKPOINT_DIR" \
    --pipeline_config_path "$CONFIG_PATH" \
    --checkpoint_steps "$CHECKPOINT_STEPS"
