#!/bin/bash
# shellcheck disable=SC2086

# Install dependecies
pip install -Ur --user requirements.txt

# Step 0: Set up experiment directory
export DATA_PREFIX="/mnt/local_storage/artifacts"
export ORIG_MODEL_NAME="CompVis/stable-diffusion-v1-4"
export ORIG_MODEL_HASH="249dd2d739844dea6a0bc7fc27b3c1d014720b28"
export ORIG_MODEL_DIR="$DATA_PREFIX/model-orig"
export ORIG_MODEL_PATH="$ORIG_MODEL_DIR/models--${ORIG_MODEL_NAME/\//--}/snapshots/$ORIG_MODEL_HASH"
export TUNED_MODEL_DIR="$DATA_PREFIX/model-tuned"

export IMAGES_REG_DIR="$DATA_PREFIX/images-reg"
export IMAGES_OWN_DIR="$DATA_PREFIX/images-own"
export IMAGES_NEW_DIR="$DATA_PREFIX/images-new"

export CLASS_NAME="car"

rm -rf $DATA_PREFIX
mkdir -p $ORIG_MODEL_DIR $TUNED_MODEL_DIR $IMAGES_REG_DIR $IMAGES_OWN_DIR $IMAGES_NEW_DIR
ln -s $DATA_PREFIX ./artifacts

# Copy own images into IMAGES_OWN_DIR
cp -rf ./data/unqtkn/*.jpg "$IMAGES_OWN_DIR/"

# Step 1: Download pretrained stable-defusion weights
python src/cache_model.py --model_dir=$ORIG_MODEL_DIR --model_name=$ORIG_MODEL_NAME --revision=$ORIG_MODEL_HASH

# Step 2: Generate regularization images with original stable-diffusion model
python src/generate.py \
  --model_dir=$ORIG_MODEL_PATH \
  --output_dir=$IMAGES_REG_DIR \
  --prompts="photo of a $CLASS_NAME" \
  --num_samples_per_prompt=200

# Step 3: Fine-tune the stable-diffusion model with Dreambooth
python src/train.py \
  --model_dir=$ORIG_MODEL_PATH \
  --output_dir=$TUNED_MODEL_DIR \
  --instance_images_dir=$IMAGES_OWN_DIR \
  --instance_prompt="a photo of unqtkn $CLASS_NAME" \
  --class_images_dir=$IMAGES_REG_DIR \
  --class_prompt="a photo of a $CLASS_NAME"

# Step 4: Inference with fine-tuned model
python src/generate.py \
  --model_dir=$TUNED_MODEL_DIR \
  --output_dir=$IMAGES_NEW_DIR \
  --prompts="photo of a unqtkn $CLASS_NAME" \
  --num_samples_per_prompt=10
