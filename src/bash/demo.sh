#!/bin/bash

GPU_ID=0
MODEL_NAME="AffNet"
TRAIN_DATA="UMD"
DATA_DIR="/mnt/storage/Tommaso/part-affordance-dataset-tools/part-affordance-dataset/data/testing"
# CKPT_PATH="/mnt/storage/Tommaso/aff-seg/checkpoints/UMD/UMD_CNN/cnn_54.pth"
# CKPT_PATH="/mnt/storage/Tommaso/aff-seg/checkpoints/UMD/UMD_DRNAtt/drnatt_44.pth"
CKPT_PATH="/mnt/storage/Tommaso/aff-seg/checkpoints/UMD/UMD_AffNet/affnet_40.pth"
# CKPT_PATH="/mnt/storage/Tommaso/aff-seg/checkpoints/UMD/UMD_Mask2Former/mask2former_89.pth"
BATCH_SIZE=1
VIS_OVERLAY=True

python src/demo.py \
--gpu_id=$GPU_ID \
--model_name=$MODEL_NAME \
--train_dataset=$TRAIN_DATA \
--data_dir=$DATA_DIR \
--checkpoint_path=$CKPT_PATH \
--batch_size=$BATCH_SIZE \
--visualise_overlay=$VIS_OVERLAY \

#--save_res=True\
#--save_overlay=True
#--dest_dir=

#python src/demo.py --gpu_id=0 --model_name=DRNAtt \
#--train_dataset=UMD \
#--data_dir=/mnt/storage/Tommaso/part-affordance-dataset-tools/part-affordance-dataset/data/testing \
#--checkpoint_path=/mnt/storage/Tommaso/aff-seg/checkpoints/UMD/UMD_DRNAtt/drnatt_44.pth \
#--batch_size=1

