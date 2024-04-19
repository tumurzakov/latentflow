import os
import torch
import logging
from einops import rearrange
import gc
import cv2
import numpy as np
from PIL import Image

from diffusers import AutoencoderKL, AutoencoderKLTemporalDecoder
from diffusers.image_processor import VaeImageProcessor
from tqdm import tqdm

from .latent_decode import LatentDecode
from .video import Video
from .latent import Latent

class VaeLatentDecode(LatentDecode):
    def __init__(self,
            vae: AutoencoderKL = None,
            vae_batch=1,
            onload_device='cuda',
            offload_device='cpu',
            progress_bar=True,
            ):
        self.vae = vae
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.vae_batch = vae_batch
        self.onload_device = onload_device
        self.offload_device = offload_device
        self.progress_bar = progress_bar

        logging.debug('VaeVideoEncode init %d, vae(%s,%s)',
                vae_batch,
                vae.device,
                vae.dtype)

    def onload(self):
        self.vae = self.vae.to(self.onload_device)

    def offload(self):
        self.vae = self.vae.to(self.offload_device)


    @torch.no_grad()
    def apply(self, latent: Latent) -> Video:
        logging.debug('VaeVideoEncode apply latent %s', latent)

        self.onload()
        latent.onload()

        v = self.decode(latent.latent)
        video = Video('HWC', video=v)

        logging.debug('VaeVideoEncode apply video %s', video)

        video.offload()
        latent.offload()
        self.offload()

        return video

    def decode(self, latents, chunk_size=100):
        video_length = latents.shape[2]
        latents = latents.to(device=self.vae.device, dtype=self.vae.dtype)
        latents = 1 / self.vae.config.scaling_factor * latents
        latents = rearrange(latents, "b c f h w -> (b f) c h w")

        kwargs = {}
        if isinstance(self.vae, AutoencoderKLTemporalDecoder):
            kwargs['num_frames'] = self.vae_batch


        rr = range(0, latents.shape[0], self.vae_batch)
        if self.progress_bar:
            rr = tqdm(rr)

        video = []
        for frame_idx in rr:
            frame = self.vae.decode(
                latents[frame_idx:frame_idx+self.vae_batch],
                **kwargs,
                ).sample
            video.append(frame.to(self.offload_device))

        # Assuming video is a list of PyTorch tensors

        # Concatenate tensors in chunks and move to CUDA
        video_cuda_list = []
        for i in range(0, len(video), chunk_size):
            chunk = torch.cat(video[i:i+chunk_size])
            video_cuda_list.append(chunk.to(self.offload_device))

        # Concatenate the chunks into a single tensor
        video = torch.cat(video_cuda_list)

        del video_cuda_list
        gc.collect()

        video = video.to(self.onload_device)

        video = rearrange(video, "(b f) c h w -> b f h w c", f=video_length)
        video.div_(2).add_(0.5).clamp_(0, 1)
        video.mul_(255)

        return video.to(self.offload_device, torch.uint8)

class VaeLatentDecodeSave(VaeLatentDecode):
    def __init__(self,
            vae: AutoencoderKL = None,
            vae_batch=1,
            save_path=None,
            save_batch_size=100,
            save_chunk_size=100,
            save_fps=16,
            start_frame=0,
            onload_device='cuda',
            offload_device='cpu',
            ):

        super().__init__(
                vae=vae,
                vae_batch=vae_batch,
                onload_device=onload_device,
                offload_device=offload_device,
                progress_bar=False,
                )

        self.save_path = save_path
        self.save_batch_size = save_batch_size
        self.save_chunk_size = save_chunk_size
        self.save_fps = save_fps
        self.start_frame = start_frame

    def apply(self, latent: Latent) -> Latent:
        self.onload()

        dirname = os.path.dirname(self.save_path)
        if not os.path.isdir(dirname):
            os.makedirs(dirname, exist_ok=True)

        if '%' in self.save_path:
            self.save_images(latent)
        else:
            self.save_video(latent)

        self.offload()

        return latent

    def save_images(self, latent):
        latents = latent.latent

        b = latents.shape[0]
        l = latents.shape[2]

        for batch_idx in range(b):
            for frame_idx in tqdm(range(0, l, self.save_batch_size)):
                ll = latents[batch_idx:batch_idx+1, :, frame_idx:frame_idx+self.save_batch_size, :, :]
                video = self.decode(ll.to(self.onload_device), chunk_size=self.save_chunk_size)
                for decode_idx, f in enumerate(video[0]):
                    frame_number = self.start_frame + frame_idx + decode_idx
                    frame = f.detach().cpu().numpy()
                    img = Image.fromarray(frame)
                    output_path = self.save_path % frame_number
                    img.save(output_path)

    def save_video(self, latent):
        latents = latent.latent

        b = latents.shape[0]
        l = latents.shape[2]
        h = latents.shape[3]*self.vae_scale_factor
        w = latents.shape[4]*self.vae_scale_factor

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # Create the VideoWriter object
        out = cv2.VideoWriter(self.save_path, fourcc, self.save_fps, (w, h))

        for batch_idx in range(b):
            for frame_idx in tqdm(range(0, l, self.save_batch_size)):
                ll = latents[batch_idx:batch_idx+1, :, frame_idx:frame_idx+self.save_batch_size, :, :]
                video = self.decode(ll.to(self.onload_device), chunk_size=self.save_chunk_size)
                for f in video[0]:
                    frame = f.detach().cpu().numpy()
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Ensure the frame is in uint8 format
                    frame = np.uint8(frame)

                    # Write the frame to the video file
                    out.write(frame)

        out.release()
