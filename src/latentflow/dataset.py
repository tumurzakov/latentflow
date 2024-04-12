import os
import sys
import torch
import logging
import random
import numpy as np
from PIL import Image
from tqdm import tqdm

from .flow import Flow
from .meta_utils import read_meta

from einops import rearrange

from transformers import CLIPTokenizer

import decord
decord.bridge.set_bridge('torch')

class VideoDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            samples_dir: str,
            video_length: int,
            tokenizer: CLIPTokenizer = None,
            width: int = -1,
            height: int = -1,
            tile_width: int = -1,
            tile_height: int = -1,
            randomize: bool = True,
            start_frame: int = 0,
            end_frame: int = -1,
    ):
        self.samples_dir = samples_dir
        self.video_length = video_length
        self.tokenizer = tokenizer
        self.width = width
        self.height = height
        self.tile_width = tile_width
        self.tile_height = tile_height
        self.randomize = randomize
        self.start_frame = start_frame
        self.end_frame = end_frame

        self.samples = []

        files = [x for x in os.listdir(samples_dir) if 'json' not in x]
        for file_name in files:
            file_path = f"{samples_dir}/{file_name}"
            name, ext = os.path.splitext(file_name)
            meta_path = f"{samples_dir}/{name}.json"
            self.samples.append((file_path, meta_path))

    def tokenize(self, prompt):
        input_ids = self.tokenizer(
            prompt,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids[0]

        return input_ids

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        file_path, meta_path = self.samples[index]
        vr = decord.VideoReader(file_path, width=self.width, height=self.height)

        start = self.start_frame
        if self.randomize and len(vr) > self.video_length:
            stop = len(vr) - self.video_length
            start = random.randint(self.start_frame, max(stop, self.end_frame-self.video_length))

        sample_index = list(range(0, len(vr)))[start:start+self.video_length]
        video = vr.get_batch(sample_index)
        video = rearrange(video, "f h w c -> f c h w")

        if self.tile_width > 0 and self.tile_height > 0:
            tile_z = start
            tile_y = random.randint(0, self.height - self.height//8)
            tile_x = random.randint(0, self.width - self.width//8)
            tile_h = random.randint(self.height//8, self.height)
            tile_w = random.randint(self.width//8, self.width)

            crop = video[
                    :,
                    :,
                    tile_y:tile_y + tile_h,
                    tile_x:tile_x + tile_w,
                    ]

            video = crop

        meta = read_meta(meta_path)

        prompt = ""
        if 'prompt' in meta:
            prompt = meta['prompt']

        if self.randomize:
            prompt = f"{prompt}, frame{start}"

        example = {}
        example['pixel_values'] = (video / 127.5 - 1.0)

        if self.tile_width > 0 and self.tile_height > 0:
            example["tile_encoding"] = self.tokenize(f"tileZ{tile_z}Y{tile_y}X{tile_x}H{tile_h}W{tile_w}")

        example["prompt_ids"] = self.tokenize(prompt)

        return example

class PrecacheVideoDataset(VideoDataset):
    def __init__(
            self,
            vae,
            text_encoder,
            cache_dir,
            precache_count=0,
            onload_device='cuda',
            offload_device='cpu',
            **kwargs
    ):
        self.vae = vae
        self.text_encoder = text_encoder
        self.cache_dir = cache_dir
        self.onload_device = onload_device
        self.offload_device = offload_device
        self.index = 0
        self.precache_count = precache_count
        super().__init__(**kwargs)

    def onload(self):
        self.vae.to(self.onload_device)
        self.text_encoder.to(self.onload_device)

    def offload(self):
        self.vae.to(self.offload_device)
        self.text_encoder.to(self.offload_device)

    def precache(self, count):
        self.precache_count = count

        self.onload()

        self.index = 0
        progress_bar = tqdm(total=count, desc=f'{self.cache_dir}')
        while self.index < count:
            for step, batch in enumerate(self):
                progress_bar.update(1)
                pass

        self.index = 0

        self.offload()

    def __getitem__(self, index):
        item = super().__getitem__(index)

        index = self.index % self.precache_count
        path = f'{self.cache_dir}/{index}.pth'
        self.index += 1

        if os.path.isfile(path):
            cache = torch.load(path)

            latents = cache['latents']

            # in case if we want to precache more frames then will use for train
            if len(latents.shape) == 4:
                latents = latents[:,:self.video_length,:,:]

            item['latents'] = latents
            item['embeddings'] = cache['embeddings']

        else:
            pixel_values = item["pixel_values"].to(self.vae.dtype)

            if len(pixel_values.shape) == 5:
                video_length = pixel_values.shape[1]
                pixel_values = rearrange(pixel_values, "b f c h w -> (b f) c h w")
                latents = self.vae.encode(pixel_values.to(self.vae.device, dtype=self.vae.dtype)).latent_dist.sample()
                latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)
            else:
                latents = self.vae.encode(pixel_values.to(self.vae.device, dtype=self.vae.dtype)).latent_dist.sample()

            latents = latents * 0.18215

            embeddings = self.text_encoder(torch.stack([item["prompt_ids"]]).to(self.text_encoder.device))[0]

            cache = {
                    'latents': latents,
                    'embeddings': embeddings,
                    }

            torch.save(cache, path)

        return item

class PhotoDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            samples_dir: str,
            tokenizer: CLIPTokenizer = None,
            height: int = None,
            width: int = None,
    ):
        self.samples_dir = samples_dir
        self.tokenizer = tokenizer
        self.height = height
        self.width = width

        self.samples = []

        files = [x for x in os.listdir(samples_dir) if not (x.endswith('json') or x.endswith('pth'))]
        for file_name in files:
            file_path = f"{samples_dir}/{file_name}"
            name, ext = os.path.splitext(file_name)
            meta_path = f"{samples_dir}/{name}.json"
            self.samples.append((file_path, meta_path))

    def tokenize(self, prompt):
        input_ids = self.tokenizer(
            prompt,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids[0]

        return input_ids

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        file_path, meta_path = self.samples[index]
        meta = read_meta(meta_path)
        image = Image.open(file_path)
        if self.width is not None and self.height is not None:
            image = image.resize((self.width, self.height))
        image = torch.tensor(np.array(image)).permute(2,0,1)

        prompt = ""
        if 'prompt' in meta:
            prompt = meta['prompt']

        example = {}
        example['pixel_values'] = (image / 127.5 - 1.0)

        extra_path = meta_path.replace('.json', '.pth')
        if os.path.isfile(extra_path):
            example["embeddings"] = torch.load(extra_path)
        else:
            example["prompt_ids"] = self.tokenize(prompt)

        return example
