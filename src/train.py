import itertools

from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.utils.import_utils import is_xformers_available
from ray.air import session, ScalingConfig
from ray.train.torch import TorchTrainer
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from transformers import CLIPTextModel

from dataset import collate, get_train_dataset
from flags import train_arguments


def prior_preserving_loss(model_pred, target, weight):
    # Chunk the noise and model_pred into two parts and compute
    # the loss on each part separately.
    model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
    target, target_prior = torch.chunk(target, 2, dim=0)

    # Compute instance loss
    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

    # Compute prior loss
    prior_loss = F.mse_loss(
        model_pred_prior.float(), target_prior.float(), reduction="mean"
    )

    # Add the prior loss to the instance loss.
    return loss + weight * prior_loss


def get_target(scheduler, noise, latents, timesteps):
    """Get the target for loss depending on the prediction type."""
    pred_type = scheduler.config.prediction_type
    if pred_type == "epsilon":
        return noise
    if pred_type == "v_prediction":
        return scheduler.get_velocity(latents, noise, timesteps)
    raise ValueError(f"Unknown prediction type {pred_type}")


def get_cuda_devices():
    devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    local_rank = session.get_local_rank()
    assert len(devices) >= 2, "Require at least 2 GPU devices to work."
    return devices[(local_rank * 2) : ((local_rank * 2) + 2)]


def load_models(config, cuda):
    """Load pre-trained Stable Diffusion models."""
    # Load all models in bfloat16 to save GRAM.
    # For models that are only used for inferencing,
    # full precision is also not required.
    dtype = torch.bfloat16

    text_encoder = CLIPTextModel.from_pretrained(
        args.model_dir,
        subfolder="text_encoder",
        torch_dtype=dtype,
    )

    noise_scheduler = DDPMScheduler.from_pretrained(
        config["model_dir"],
        subfolder="scheduler",
        torch_dtype=dtype,
    )

    # VAE is only used for inference, keeping weights in full precision is not required.
    vae = AutoencoderKL.from_pretrained(
        config["model_dir"],
        subfolder="vae",
        torch_dtype=dtype,
    )

    # Convert unet to bf16 to save GRAM.
    unet = UNet2DConditionModel.from_pretrained(
        config["model_dir"],
        subfolder="unet",
        torch_dtype=dtype,
    )
    if is_xformers_available():
        unet.enable_xformers_memory_efficient_attention()

    # Move the models to target devices
    # UNET is the largest component, occupying the first GPU by itself.
    unet.to(cuda[0])
    unet.train()

    # Fine-tune the text encoder
    text_encoder.to(cuda[1])
    text_encoder.train()

    # Freeze the VAE image encoder
    vae.requires_grad_(False)
    vae.to(cuda[1])

    torch.cuda.empty_cache()

    return text_encoder, noise_scheduler, vae, unet


def train_fn(config):
    """The training loop for each worker."""

    cuda = get_cuda_devices()

    # Load pre-trained models with device mapping
    # GPU 0: Unet
    # GPU 1: VAE, Text Encoder
    text_encoder, noise_scheduler, vae, unet = load_models(config, cuda)

    # Wrap the models for DistributedDataParallel training
    unet = DDP(unet, device_ids=[cuda[0]], output_device=cuda[0])
    text_encoder = DDP(text_encoder, device_ids=[cuda[1]], output_device=cuda[1])

    # Define an AdamW optimizer
    optimizer = torch.optim.AdamW(
        itertools.chain(text_encoder.parameters(), unet.parameters()),
        lr=config["lr"],
    )

    # Train!
    num_epochs = config["num_epochs"]
    batch_size = config["train_batch_size"]

    # Fetch the sharded Ray Dataset
    train_dataset = session.get_dataset_shard("train")

    global_step = 0
    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs} starts...")
        data_iterator = train_dataset.iter_torch_batches(
            batch_size=batch_size, device=cuda[1]
        )

        for batch in data_iterator:
            # Load batch on GPU 2 because VAE and text encoder are there.
            batch = collate(batch, cuda[1], torch.bfloat16)

            optimizer.zero_grad()

            # Encode the input images
            latents = vae.encode(batch["images"]).latent_dist.sample() * 0.18215

            # Sample noise and a random timestep for each image
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (batch_size,),
                device=latents.device,
            ).long()

            # Add noise to the latents according to the noise magnitude at each timestep
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Get the text embedding for conditioning
            encoder_hidden_states = text_encoder(batch["prompt_ids"])[0]

            # Predict the noise residual. Move all tensors to GPU 0 where unet resides.
            model_pred = unet(
                noisy_latents.to(cuda[0]),
                timesteps.to(cuda[0]),
                encoder_hidden_states.to(cuda[0]),
            ).sample
            target = get_target(noise_scheduler, noise, latents, timesteps).to(cuda[0])

            # Now, move model prediction to GPU 2 for loss calculation.
            loss = prior_preserving_loss(
                model_pred, target, config["prior_loss_weight"]
            )
            loss.backward()

            # Gradient clipping before optimizer stepping.
            clip_grad_norm_(
                itertools.chain(text_encoder.parameters(), unet.parameters()),
                config["max_grad_norm"],
            )

            optimizer.step()  # Step all optimizers.

            global_step += 1
            results = {
                "step": global_step,
                "loss": loss.detach().item(),
            }
            session.report(results)

    # Save the model checkpoint after fine-tuning finished 
    if session.get_world_rank() == 0:
        pipeline = DiffusionPipeline.from_pretrained(
            config["model_dir"],
            text_encoder=text_encoder.module,
            unet=unet.module,
        )
        pipeline.save_pretrained(config["output_dir"])


if __name__ == "__main__":
    args = train_arguments().parse_args()

    # Build training dataset.
    train_dataset = get_train_dataset(args)

    print(f"Loaded training dataset (size: {train_dataset.count()})")

    # Train with Ray AIR TorchTrainer.
    trainer = TorchTrainer(
        train_fn,
        train_loop_config=vars(args),
        scaling_config=ScalingConfig(
            use_gpu=True,
            num_workers=args.num_workers,
            resources_per_worker={"GPU": 2},
        ),
        datasets={
            "train": train_dataset,
        },
    )
    result = trainer.fit()

    print(result)
