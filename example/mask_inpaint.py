from einops import rearrange

from latentflow.state import State
from latentflow.video import Video
from latentflow.latent import Latent
from latentflow.tile import Tile
from latentflow.prompt import Prompt
from latentflow.vae_video_encode import VaeVideoEncode
from latentflow.vae_latent_decode import VaeLatentDecode
from latentflow.diffuse import Diffuse
from latentflow.unet import Unet
from latentflow.prompt_embeddings import PromptEmbeddings
from latentflow.a1111_prompt_encode import A1111PromptEncode
from latentflow.compel_prompt_encode import CompelPromptEncode
from latentflow.video_load import VideoLoad
from latentflow.video_show import VideoShow
from latentflow.latent_show import LatentShow
from latentflow.noise import Noise
from latentflow.add_noise import AddNoise
from latentflow.schedule import Schedule
from latentflow.debug import Debug, DebugHash
from latentflow.invert import Invert
from latentflow.bypass import Bypass
from latentflow.lora import LoraOn, LoraOff
from latentflow.apply import Apply
from latentflow.controlnet import ControlNet
from latentflow.loop import Loop
from latentflow.tile import Tile, TileGenerator
from latentflow.step import Step
from latentflow.tensor import Tensor,TensorAdd
from latentflow.mask import MaskEncode, Mask, LatentMaskCut, VideoMaskCut, LatentMaskMerge, LatentMaskCrop
from latentflow.region import Region
from latentflow.noop import Noop
from latentflow.video_vae_latent_decode import VideoVaeLatentDecode
from latentflow.interpolate import Interpolate
from latentflow.flow import If, Set
from latentflow.nn_latent_upscale import NNLatentUpscale

import torch
torch.backends.cuda.matmul.allow_tf32 = True

unet = Unet(
    unet=pipe.unet.to('cuda'), 
    scheduler=pipe.scheduler,
    fp16=True,
)

cnet = ControlNet(
    controlnet=pipe.controlnet.to('cuda'),
    scheduler=pipe.scheduler,
)



state = State({
    'width': 512,
    'height': 288,
    'video_length': 48,
    'start_frame': 32,
    'num_inference_steps': 10,
})

video = \
    (Debug("Video to video")
        | VideoLoad(f'{root}/pipelines/clip1_couplet1_6/data/input/in.mp4', 
                    device='cuda', 
                    video_length=state['video_length'],
                    width=state['width'], 
                    height=state['height'],
                    start_frame=state['start_frame'],
                   )
        - VideoShow(fps=16)
        | VaeVideoEncode(
            vae=pipe.vae.to('cuda'),
            cache=f'{root}/pipelines/clip1_couplet1_6/data/infer/latents.pth',
            video_length=state['video_length'],
            )
        - LatentShow(fps=16, vae=vae)
        | Invert(
            tokenizer=pipe.tokenizer, 
            text_encoder=pipe.text_encoder, 
            unet=pipe.unet, 
            scheduler=pipe.scheduler,
            cache=f'{root}/pipelines/clip1_couplet1_6/data/infer/inv_latents.pth',
            )
        - LatentShow(fps=16, vae=pipe.vae.to('cuda'))
        > state("latent")
        ) >> \
    (Debug("Loading facemask")
      | VideoLoad([
              f'{root}/pipelines/clip1_couplet1_6/data/masks/face_mask_288.mp4',
          ], device='cuda', 
                  video_length=state['video_length'], 
                  width=state['width'], 
                  height=state['height'],
                  start_frame=state['start_frame'],
                 )
      - VideoShow(fps=16)
      > state('face_mask_video')
      ) >> \
    (state['face_mask_video'] 
      | MaskEncode(width=state['width'], height=state['height'])
      > state('face_mask')
      ) >> \
    (state['face_mask_video'] 
      | MaskEncode(width=state['width']//8, height=state['height']//8, channels=4)
      > state('latent_face_mask')
      ) >> \
    (Debug("Loading source")
      | VideoLoad([
              f'{root}/pipelines/clip1_couplet1_6/data/face_source.mp4',
          ], 
          device='cuda', 
          video_length=state['video_length'],
          start_frame=state['start_frame'],
      )
      - VideoShow(fps=16)
      | VaeVideoEncode(
            vae=pipe.vae.to('cuda'),
            cache=f'{root}/pipelines/clip1_couplet1_6/data/infer/source.pth',
            video_length=state['video_length'],
            )
      > state('face_source')
      ) >> \
    (Debug("Loading controlnet")
      | VideoLoad([
              f'{root}/pipelines/clip1_couplet1_6/data/lineart_out/lineart.mp4',
          ], 
          device='cuda', 
          video_length=state['video_length'],
          start_frame=state['start_frame'],
          )
      - VideoShow(fps=16)
      > state('controlnet_image')
      ) >> \
    (Prompt(prompt=
                "(young woman in blue haute couture dress)1.0"
                "(woman sits behind table)1.2"
                "(office employees working behind)1.0"
                "(blue)0.5, cyberpunk, fantasy, sci-fi, stunning, concept, artstation, acid colors,"
                "octane render, Unreal engine, cg unity, hdr, wallpaper, neonpunkai, lineart",
            negative_prompt=
                "deformed, distorted, disfigured)1.0,"
                "poorly drawn, bad anatomy, wrong anatomy,"
                "extra limb,missing limb, floating limbs,"
                "(mutated hands and fingers)1.0,"
                "disconnected limbs, mutation, mutated,"
                "ugly, disgusting, blurry, amputation,",
            frames=list(range(0, state['video_length']))
           )
        | CompelPromptEncode(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder.to('cuda'))
        | Region(
            latent=state['latent'].clone(),
            scheduler=type(pipe.scheduler).from_config(pipe.scheduler.config),
            guidance_scale=10,
            controlnet_image=state['controlnet_image'].resize((state['video_length'], state['height'], state['width'])),
            controlnet_scale=[1.0],
            )
        > state(('regions', 1))
        ) >> \
    (state['latent_face_mask'].resize(state['face_source'].latent.shape[2:])
        | LatentMaskCrop(width=state['width']//8, height=state['height']//8)
        > state('latent_mask_crop')
    ) >> \
    (Prompt(prompt=
                "(young woman in red haute couture dress)1.0"
                "(woman sits behind table)1.2"
                "(office employees working behind)1.0"
                "(blue)0.5, cyberpunk, fantasy, sci-fi, stunning, concept, artstation, acid colors,"
                "octane render, Unreal engine, cg unity, hdr, wallpaper, neonpunkai, lineart",
            negative_prompt=
                "deformed, distorted, disfigured)1.0,"
                "poorly drawn, bad anatomy, wrong anatomy,"
                "extra limb,missing limb, floating limbs,"
                "(mutated hands and fingers)1.0,"
                "disconnected limbs, mutation, mutated,"
                "ugly, disgusting, blurry, amputation,",
            frames=list(range(0, state['video_length']))
           )
        | CompelPromptEncode(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder.to('cuda'))
        | Region(
            latent=state['face_source']
                | LatentMaskCut(latent_mask_crop=state['latent_mask_crop'])
                - LatentShow(fps=16, vae=vae, vae_batch=1)
                | Invert(
                    tokenizer=pipe.tokenizer, 
                    text_encoder=pipe.text_encoder, 
                    unet=pipe.unet, 
                    scheduler=pipe.scheduler,
                    cache=f'{root}/pipelines/clip1_couplet1_6/data/infer/region1_inv_latents.pth',
                    ),
            scheduler=type(pipe.scheduler).from_config(pipe.scheduler.config),
            guidance_scale=10,
            controlnet_image=state['controlnet_image']
                - VideoShow(fps=16)    
                | VideoMaskCut(state['face_mask'], width=state['width'], height=state['height'])
                - VideoShow(fps=16),
            controlnet_scale=[1.0],
            mask=state['latent_face_mask'],
            )
        > state(('regions', 0))
        ) >> \
    (state | Loop(state['regions'], name='Region loop', callback=lambda region_index, region:
        (state | Schedule(region.scheduler, num_inference_steps=state['num_inference_steps']))
    )) >> \
    (state | Schedule(pipe.scheduler, num_inference_steps=state['num_inference_steps']) > state('timesteps')) >> \
    (state | Loop(state['timesteps'], name="Denoise loop", callback=lambda timestep_index, timestep:
        (state | Loop(state['regions'], name='Region loop', callback=lambda region_index, region:
            (state
                | cnet(
                    timestep_index, 
                    timestep, 
                    latent=region.latent, 
                    image=region.controlnet_image.cnet(),
                    controlnet_scale=region.controlnet_scale,
                    timesteps=state['timesteps'],
                    embeddings=region.embeddings,
                ) 
                | unet(
                    timestep=timestep, 
                    latent=region.latent,
                    embeddings=region.embeddings,
                    guidance_scale=region.guidance_scale,
                    scheduler=region.scheduler,
                )
                | Step(
                    scheduler=region.scheduler,
                    timestep=timestep, 
                    latent=region.latent
                )
                - LatentShow(fps=16, vae=pipe.vae.to('cuda'))
                | Set(region.latent)) >> \
            (state 
                | If(
                    lambda : region_index==0, 
                    if_true=LatentMaskMerge(
                        background=state['regions'][1].latent, 
                        foreground=state['latent_mask_crop'].restore(state['regions'][0].latent),
                        mask=state['regions'][0].mask
                    ), 
                    if_false=state['regions'][1].latent
                ) 
                | Set(state['regions'][1].latent))
            )
        ) | Debug("End region loop")
    ) | Debug("End denoise loop")) >> \
    (state | Loop(state['regions'], name='Region loop', callback=lambda region_index, region:
        region.latent|LatentShow(fps=16, vae=vae)
    ))