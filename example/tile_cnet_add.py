from einops import rearrange

from latentflow.state import State
from latentflow.video import Video
from latentflow.latent import Latent, LatentAdd, NoisePredict
from latentflow.tile import Tile
from latentflow.prompt import Prompt
from latentflow.vae_video_encode import VaeVideoEncode
from latentflow.vae_latent_decode import VaeLatentDecode
from latentflow.diffuse import Diffuse
from latentflow.unet import Unet, UnetNoCFG, DoCFG
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
from latentflow.slice import slice_scale

import torch
torch.backends.cuda.matmul.allow_tf32 = True

unet = UnetNoCFG(
    unet=pipe.unet.to('cuda'),
    scheduler=pipe.scheduler,
    guidance_scale=10,
    fp16=True,
)

cnet = ControlNet(
    controlnet=pipe.controlnet.to('cuda'),
    scheduler=pipe.scheduler,
)

state = State({
    'width': 512,
    'height': 288,
    'video_length': 96,

    'tile': {'length':48,'height':288//2,'width':512//2},

    'vae_batch': 4,
    'num_inference_steps': 10,
    'guidance_scale': 10,
})

video = \
    (Debug("Video to video")
        | VideoLoad(f'{data}/input/in.mp4', device='cuda', video_length=state['video_length'], start_frame=32, width=512, height=288)
        - VideoShow(fps=16)
        | VaeVideoEncode(
            vae=pipe.vae.to('cuda'),
            cache=f'{data}/infer/latents.pth',
            video_length=state['video_length'],
            )
        - LatentShow(fps=16, vae=pipe.vae.to('cuda'))
        | Invert(
            tokenizer=pipe.tokenizer,
            text_encoder=pipe.text_encoder,
            unet=pipe.unet,
            scheduler=pipe.scheduler,
            cache=f'{data}/infer/inv_latents.pth',
            temporal_context=state['tile']['length'],
            )
        - LatentShow(fps=16, vae=vae, vae_batch=state['vae_batch'])
        > state("latent")
        ) >> \
    (Latent(shape=(1,4,state['video_length'],state['height']//8,state['width']//8), device='cuda')
        | Noise(scheduler=pipe.scheduler, device='cuda')
        - LatentShow(fps=16, vae=vae, vae_batch=state['vae_batch'])
        < state("latent")
        ) >> \
    (Prompt(prompt=
               "(young woman in haute couture dress)1.0,"
            negative_prompt=
               "deformed, distorted, disfigured)1.0,"
               "poorly drawn, bad anatomy, wrong anatomy,"
               "extra limb,missing limb, floating limbs,"
               "(mutated hands and fingers)1.0,"
               "disconnected limbs, mutation, mutated,"
               "ugly, disgusting, blurry, amputation,",
            frames=list(range(0,state['video_length'])),
           )
        | CompelPromptEncode(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder.to('cuda'))
        > state("embeddings")
        ) >> \
    (Debug("Loading controlnet")
      | VideoLoad([
              f'{data}/lineart_out/lineart.mp4',
          ], device='cuda', video_length=state['video_length'], width=state['width'], height=state['height'])
      - VideoShow(fps=16)
      > state('controlnet_image')
      ) >> \
    (state \
        | Schedule(scheduler=pipe.scheduler, num_inference_steps=state['num_inference_steps'])
        > state('timesteps')
        ) >> \
    (state['latent']
        - LatentShow(fps=16, vae=vae)
        - LoraOn(loras={
            f"{models}/lora/details.safetensors": 0.8,
            }, pipe=pipe)
        | Loop(state['timesteps'], name="Denoise loop", callback=lambda timestep_index, timestep:
            (Latent(latent=torch.zeros_like(state['latent'].latent).repeat(1,1,1,1,1)) > state('noise_predict')) >> \
            (Tensor(torch.zeros_like(state['latent'].latent)) > state('pixel_infer_count')) >> \
            (state
                 | Loop(
                    TileGenerator(
                        Tile(
                            height=state['tile']['height']//8,
                            width=state['tile']['width']//8,
                            length=state['tile']['length'],
                            length_overlap=state['tile']['length']//2,
                            height_overlap=state['tile']['height']//8//2,
                            width_overlap=state['tile']['width']//8//2,
                            ),
                        state['latent'],
                        strategy='simple',
                        #random_threshold=0.8,
                        pixel_infer_count=state['pixel_infer_count'],
                    ),
                    name="Tile loop",
                    callback=lambda tile_index, tile:
                        (Tensor(torch.zeros((1,4,state['tile']['length'],state['tile']['height']//8,state['tile']['width']//8), device='cuda'))
                            | TensorAdd(state['latent'].latent[tile])
                            | Apply(lambda x: Latent(x.tensor))
                            | Debug("TileLatent")
                            > state("tile_latent")) >> \

                        (Tensor(torch.zeros((1,state['tile']['length'],3,state['tile']['height'],state['tile']['width']), device='cuda'))
                            | Debug("Controlnet tile tensor")
                            | TensorAdd((state['controlnet_image'].cnet())[:,tile[2],:,slice_scale(tile[3],8),slice_scale(tile[4],8)])
                            - Debug("TileControlNet", lambda x: x.tensor.shape)
                            > state("tile_controlnet_image")) >> \

                        state
                            | cnet(
                                timestep_index=timestep_index,
                                timestep=timestep,
                                latent=state["tile_latent"],
                                image=state["tile_controlnet_image"].tensor,
                                timesteps=state['timesteps'],
                                embeddings=state['embeddings'].slice(slice_scale(tile[2],2)),
                                controlnet_scale=[1.0],
                            )
                            | unet(
                                timestep=timestep,
                                latent=state['tile_latent'],
                                embeddings=state['embeddings'].slice(slice_scale(tile[2],2)),
                                guidance_scale=state['guidance_scale'],
                            )
                            | Apply(lambda l: Latent(latent=l.latent[
                                      0: state['tile_latent'].shape[0],
                                      0: state['tile_latent'].shape[1],
                                      0: state['tile_latent'].shape[2],
                                      0: state['tile_latent'].shape[3],
                                      0: state['tile_latent'].shape[4],
                                    ]))
                            | LatentAdd(Latent(state['noise_predict'].latent[tile]))

                  ) | Debug("Tile loop end")
            ) >> \
            (state['noise_predict']
               - LatentShow(fps=16, vae=vae, vae_batch=state['vae_batch'])
               | Apply(lambda x: NoisePredict(x.latent/state['pixel_infer_count'].tensor.repeat(2,1,1,1,1)))
               - LatentShow(fps=16, vae=vae, vae_batch=state['vae_batch'])
               | DoCFG(guidance_scale=state['guidance_scale'])
               - LatentShow(fps=16, vae=vae, vae_batch=state['vae_batch'])
               | Step(
                   scheduler=pipe.scheduler,
                   timestep=timestep,
                   latent=state['latent'])
               | LatentShow(fps=16, vae=vae, vae_batch=state['vae_batch'])
               > state("latent")
            )
        )
        | Debug("Denoise loop end")
    ) >> \
    (state['latent'] | VideoVaeLatentDecode(vae=vae, vae_batch=state['vae_batch']) | VideoShow(fps=16) )

