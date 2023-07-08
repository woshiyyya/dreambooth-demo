#!/bin/bash

# Set Environment Variables
export TF_CPP_MIN_LOG_LEVEL="3"
export DATA_SYMLINK="./artifacts"
export DATA_PREFIX="/mnt/local_storage/artifacts"
export TUNED_MODEL_DIR="$DATA_PREFIX/model-tuned"
export IMAGES_NEW_DIR="$DATA_PREFIX/images-new"
export CLASS_NAME="car"

# Replace it with your own model checkpoint
export TUNED_MODEL_S3_URI="s3://demo-pretrained-model/dreambooth-stable-diffusion-finetuned/" 


if [ -e $DATA_SYMLINK ]; then
  rm -f $DATA_SYMLINK
fi

if [ -e $DATA_PREFIX ]; then
  rm -rf $DATA_PREFIX
fi

mkdir -p $TUNED_MODEL_DIR $IMAGES_NEW_DIR
ln -s $DATA_PREFIX $DATA_SYMLINK

# AWS CLI configurations to speed up ckpt downloads
awsv2 configure set s3.max_concurrent_requests 32
awsv2 configure set default.s3.preferred_transfer_client crt
awsv2 configure set default.s3.target_bandwidth 100Gb/s
awsv2 configure set default.s3.multipart_chunksize 8MB

# Download the tuned model checkpoint
awsv2 s3 sync $TUNED_MODEL_S3_URI $TUNED_MODEL_DIR > /dev/null

# Generate images with prompts
python src/generate.py \
  --model_dir=$TUNED_MODEL_DIR \
  --output_dir=$IMAGES_NEW_DIR \
  --prompts="photo of a unqtkn $CLASS_NAME in the beach" \
  --num_samples_per_prompt=10