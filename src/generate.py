""" 
Script for generating images using the stable diffusion model

Exapmle usage:
python src/generate.py \
  --model_dir=$MODEL_CHECKPOINT_DIR \
  --output_dir=$IMAGES_OUTPUT_DIR \
  --prompts="photo of a unqtkn car in the woods" \
  --num_samples_per_prompt=10
"""

import os
import torch
import random
import hashlib
import numpy as np
from os import path
from flags import run_model_flags
from diffusers import DiffusionPipeline

import warnings
warnings.filterwarnings("ignore")

def seed_everything(seed):
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def run(args):
    seed_everything(888)

    print(f"Loading model from {args.model_dir}")
    pipeline = DiffusionPipeline.from_pretrained(
        args.model_dir, torch_dtype=torch.float16
    )
    pipeline.set_progress_bar_config(disable=True)
    if torch.cuda.is_available():
        pipeline.to("cuda")

    prompts = args.prompts.split(",")

    # Generate 1 image to reduce memory consumption.
    for prompt in prompts:
        for i in range(args.num_samples_per_prompt):
            for image in pipeline(prompt).images:
                hash_image = hashlib.sha1(image.tobytes()).hexdigest()
                image_filename = path.join(args.output_dir, f"{i}-{hash_image}.jpg")
                image.save(image_filename)
                print(f"Saved new image: {image_filename}")


if __name__ == "__main__":
    args = run_model_flags().parse_args()
    run(args)
