import torch
import logging
import math
import sys

from .flow import Flow
from .meta_utils import read_meta
from .debug import tensor_hash

from diffusers.optimization import get_scheduler
from diffusers.utils import convert_state_dict_to_diffusers
from diffusers import StableDiffusionPipeline
from diffusers.utils.torch_utils import is_compiled_module
from peft.utils import get_peft_model_state_dict

from accelerate import Accelerator
from einops import rearrange
import torch.nn.functional as F

from tqdm import tqdm

class Params(Flow):
    def __init__(self, params):
        self.params = params

    def __str__(self):
        count = sum(p.numel() for p in self.params if p.requires_grad)
        tensors = []
        for p in self.params:
            tensors.append(p.data)
        return "Params(%s, %s)" % (count, tensor_hash(tensors))

class TrainConfig:
    train_batch_size = 1
    gradient_accumulation_steps = 1
    mixed_precision = 'fp16'
    report_to = None
    learning_rate = 1e-4
    max_grad_norm = 1.0
    use_optimizer = 'AdamW'

    adam_beta1=0.9
    adam_beta2=0.99
    adam_weight_decay = 1e-2
    adam_epsilon = 1e-08

    lr_scheduler = 'constant'
    lr_warmup_steps = 0

    first_epoch = 0
    num_train_epochs = 1
    max_train_steps = 1

    output_dir = None
    checkpointing_steps = None

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class Train(Flow):
    def __init__(self,
            pipe,
            dataset,
            train_config = None,
            onload_device='cuda',
            offload_device='cpu',
            do_offload_text_encoder=False,
            do_offload_vae=False,
            ):

        self.pipe = pipe
        self.train_dataset = dataset
        self.train_config = train_config
        self.onload_device = onload_device
        self.offload_device = offload_device
        self.do_offload_text_encoder = do_offload_text_encoder
        self.do_offload_vae = do_offload_vae

        self.unet = pipe.unet
        self.vae = pipe.vae

        if train_config is None:
            self.train_config = TrainConfig()

        self.train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.train_config.train_batch_size
        )

        self.noise_scheduler = self.pipe.scheduler
        self.text_encoder = self.pipe.text_encoder

    def onload(self):
        self.unet.to(self.onload_device)

    def offload(self):
        self.unet.to(self.offload_device)

    def onload_text_encoder(self):
        self.text_encoder.to(self.onload_device)

    def offload_text_encoder(self):
        self.text_encoder.to(self.offload_device)

    def onload_vae(self):
        self.vae.to(self.onload_device)

    def offload_vae(self):
        self.vae.to(self.offload_device)

    def apply(self, params):
        self.onload()

        def unwrap_model(model):
            model = accelerator.unwrap_model(model)
            model = model._orig_mod if is_compiled_module(model) else model
            return model

        params_to_optimize = params.params

        accelerator = Accelerator(
            gradient_accumulation_steps=self.train_config.gradient_accumulation_steps,
            mixed_precision=self.train_config.mixed_precision,
            log_with=self.train_config.report_to,
        )

        # Initialize the optimizer
        optimizer = None
        if self.train_config.use_optimizer == 'AdamW8bit':
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
                )

            optimizer_cls = bnb.optim.AdamW8bit
        elif self.train_config.use_optimizer == 'Prodigy':
            try:
                import prodigyopt
            except ImportError:
                raise ImportError(
                    "Please install prodigyopt. You can do so by running `pip install prodigyopt`"
                )

            # https://rentry.org/59xed3
            optimizer = prodigyopt.Prodigy(
                params_to_optimize,
                lr=1.,
                decouple=True,
                weight_decay=0.01,
                d_coef=2,
                use_bias_correction=True,
                safeguard_warmup=True,
            )
        else:
            optimizer_cls = torch.optim.AdamW

        if optimizer == None:
            optimizer = optimizer_cls(
                params_to_optimize,
                lr=float(self.train_config.learning_rate),
                betas=(self.train_config.adam_beta1, self.train_config.adam_beta2),
                weight_decay=self.train_config.adam_weight_decay,
                eps=self.train_config.adam_epsilon,
            )

        lr_scheduler = get_scheduler(
            self.train_config.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=self.train_config.lr_warmup_steps * self.train_config.gradient_accumulation_steps,
            num_training_steps=self.train_config.max_train_steps * self.train_config.gradient_accumulation_steps,
        )

        global_step = 0

        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        num_update_steps_per_epoch = math.ceil(len(self.train_dataloader) / self.train_config.gradient_accumulation_steps)
        # Afterwards we recalculate our number of training epochs
        self.train_config.num_train_epochs = math.ceil(self.train_config.max_train_steps / num_update_steps_per_epoch)

        progress_bar = tqdm(range(0, self.train_config.max_train_steps), disable=not accelerator.is_local_main_process)
        progress_bar.set_description("Steps")

        for epoch in range(self.train_config.first_epoch, self.train_config.num_train_epochs):
            self.unet.train()
            train_loss = 0.0
            for step, batch in enumerate(self.train_dataloader):
                with accelerator.accumulate(self.unet):

                    if 'latents' in batch:
                        latents = batch['latents']

                    else:
                        self.onload_vae()

                        # Convert videos to latent space
                        pixel_values = batch["pixel_values"].to(self.unet.dtype)

                        if len(pixel_values.shape) == 5:
                            video_length = pixel_values.shape[1]
                            pixel_values = rearrange(pixel_values, "b f c h w -> (b f) c h w")
                            latents = self.vae.encode(pixel_values.to(self.vae.device, dtype=self.vae.dtype)).latent_dist.sample()
                            latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)
                        else:
                            latents = self.vae.encode(pixel_values.to(self.vae.device, dtype=self.vae.dtype)).latent_dist.sample()

                        latents = latents * 0.18215

                        if self.do_offload_vae:
                            self.offload_vae()

                    # Sample noise that we'll add to the latents
                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0]
                    # Sample a random timestep for each video
                    timesteps = torch.randint(0, self.noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
                    timesteps = timesteps.long()

                    # Add noise to the latents according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)


                    # Get the text embedding for conditioning
                    if "embeddings" in batch:
                        embeddings = batch["embeddings"]
                        if len(embeddings.shape) == 4:
                            embeddings = embeddings[0]
                        if embeddings.shape[0] == 2:
                            embeddings = embeddings[1:2,...]
                        encoder_hidden_states = embeddings
                    else:
                        self.onload_text_encoder()
                        encoder_hidden_states = self.text_encoder(batch["prompt_ids"].to(self.text_encoder.device))[0]

                        if "tile_encoding" in batch:
                            tile_cond = self.text_encoder(batch["tile_encoding"].to(self.text_encoder.device))[0]
                            cond = torch.cat([encoder_hidden_states, tile_cond], dim=1)

                        if self.do_offload_text_encoder:
                            self.offload_text_encoder()

                    # Get the target for loss depending on the prediction type
                    if self.noise_scheduler.prediction_type == "epsilon":
                        target = noise
                    elif self.noise_scheduler.prediction_type == "v_prediction":
                        target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        raise ValueError(f"Unknown prediction type {self.noise_scheduler.prediction_type}")

                    # Predict the noise residual and compute loss
                    model_pred = self.unet(
                            noisy_latents.to(self.unet.device),
                            timesteps,
                            encoder_hidden_states.to(self.unet.device)).sample

                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                    # Gather the losses across all processes for logging (if we use distributed training).
                    avg_loss = accelerator.gather(loss.repeat(self.train_config.train_batch_size)).mean()
                    train_loss += avg_loss.item() / self.train_config.gradient_accumulation_steps

                    # Backpropagate
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(params_to_optimize, self.train_config.max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                if accelerator.sync_gradients:
                    global_step += 1
                    progress_bar.update(1)
                    train_loss = 0.0

                    if self.train_config.checkpointing_steps is not None \
                        and global_step % self.train_config.checkpointing_steps == 0 \
                        and self.train_config.output_dir is not None \
                        and accelerator.is_main_process:

                        save_path = f'{self.train_config.output_dir}/checkpoint-{global_step}'
                        unwrapped_unet = unwrap_model(self.unet)
                        unet_lora_state_dict = convert_state_dict_to_diffusers(
                            get_peft_model_state_dict(unwrapped_unet)
                        )

                        StableDiffusionPipeline.save_lora_weights(
                            save_directory=save_path,
                            unet_lora_layers=unet_lora_state_dict,
                            safe_serialization=True,
                        )

                    if global_step >= self.train_config.max_train_steps:
                        break

        save_path = f'{self.train_config.output_dir}'
        unwrapped_unet = unwrap_model(self.unet)
        unet_lora_state_dict = convert_state_dict_to_diffusers(
            get_peft_model_state_dict(unwrapped_unet)
        )

        StableDiffusionPipeline.save_lora_weights(
            save_directory=save_path,
            unet_lora_layers=unet_lora_state_dict,
            safe_serialization=True,
        )

        self.offload()

        return params
