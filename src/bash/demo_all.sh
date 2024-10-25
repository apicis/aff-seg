#!/bin/bash

GPU_ID=0
BATCH_SIZE=1
VIS_OVERLAY=False
SAVE_RES=True
SAVE_OVERLAY=True
TRAIN_DATA="UMD"
DATA_DIR="/mnt/storage/Tommaso/part-affordance-dataset-tools/part-affordance-dataset/data/testing"
DEST_DIR_BASE="/mnt/storage/Tommaso/affseg_release_test/UMD/"
MODEL_NAME="CNN"
for MODEL_NAME in "CNN" "DRNAtt" "AffNet" "Mask2Former"
do
  # echo $MODEL_NAME
  if [ "$MODEL_NAME" = "CNN" ]
  then
    # echo "first"
    CKPT_PATH="/mnt/storage/Tommaso/aff-seg/checkpoints/UMD/UMD_CNN/cnn_54.pth"
  elif [[ "$MODEL_NAME" == "DRNAtt" ]]
  then
    # echo "second"
    CKPT_PATH="/mnt/storage/Tommaso/aff-seg/checkpoints/UMD/UMD_DRNAtt/drnatt_44.pth"
  elif [[ "$MODEL_NAME" == "AffNet" ]]
  then
    # echo "third"
    CKPT_PATH="/mnt/storage/Tommaso/aff-seg/checkpoints/UMD/UMD_AffNet/affnet_40.pth"
  elif [[ "$MODEL_NAME" == "Mask2Former" ]]
  then
    #echo "fourth"
    CKPT_PATH="/mnt/storage/Tommaso/aff-seg/checkpoints/UMD/UMD_Mask2Former/mask2former_89.pth"
  fi
  # echo "launch script"
  DEST_DIR="$DEST_DIR_BASE$MODEL_NAME"
#  echo "$DEST_DIR"
# python src/demo.py --gpu_id=$GPU_ID --model_name=$MODEL_NAME --train_dataset=$TRAIN_DATA --data_dir=$DATA_DIR --checkpoint_path=$CKPT_PATH --batch_size=$BATCH_SIZE --save_res=$SAVE_RES --save_overlay=$SAVE_OVERLAY --dest_dir=$DEST_DIR
  python src/demo.py \
  --gpu_id=$GPU_ID \
  --model_name=$MODEL_NAME \
  --train_dataset=$TRAIN_DATA \
  --data_dir=$DATA_DIR \
  --checkpoint_path=$CKPT_PATH \
  --batch_size=$BATCH_SIZE \
  --save_overlay=$SAVE_OVERLAY \
  --save_res=$SAVE_RES \
  --dest_dir=$DEST_DIR
done
