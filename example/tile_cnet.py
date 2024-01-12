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

import torch
torch.backends.cuda.matmul.allow_tf32 = True

def scale_slice(s, q):
    return slice(s.start*q, s.stop*q)

unet = Unet(
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
    'width': 1024,
    'height': 576,
    'video_length': 96,
})

video = \
    (Debug("Video to video")
        - VideoLoad(f'input/in.mp4', device='cuda', video_length=state['video_length'])
        - VideoShow(fps=16)
        - VaeVideoEncode(
            vae=pipe.vae.to('cuda'),
            cache=f'infer/latents.pth',
            video_length=state['video_length'],
            )
        - LatentShow(fps=16, vae=pipe.vae.to('cuda'))
        - Invert(
            tokenizer=pipe.tokenizer,
            text_encoder=pipe.text_encoder,
            unet=pipe.unet,
            scheduler=pipe.scheduler,
            cache=f'infer/inv_latents.pth',
            )
        - LatentShow(fps=16, vae=pipe.vae.to('cuda'))
        < state("latent")
        ) >> \
    (Latent(latent=torch.randn((1,4,state['video_length'],state['height']//8,state['width']//8)), device='cuda')
        | Noise(scheduler=pipe.scheduler)
        - LatentShow(fps=16, vae=pipe.vae.to('cuda'))
        > state("latent")
        ) >> \
    (Prompt(prompt=
               "(young woman in red haute couture dress)1.0,"
            negative_prompt=
               "(red)0.5"
               "deformed, distorted, disfigured)1.0,"
               "poorly drawn, bad anatomy, wrong anatomy,"
               "extra limb,missing limb, floating limbs,"
               "(mutated hands and fingers)1.0,"
               "disconnected limbs, mutation, mutated,"
               "ugly, disgusting, blurry, amputation,"
           )
        | CompelPromptEncode(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder.to('cuda'))
        > state("embeddings")
        ) >> \
    (Debug("Loading controlnet")
      | VideoLoad([
              f'lineart_out/lineart.mp4',
          ], device='cuda', video_length=state['video_length'], width=1024, height=576)
      - VideoShow(fps=16)
      > state('controlnet_image')
      ) >> \
    (state \
        | Schedule(scheduler=pipe.scheduler, num_inference_steps=8)
        > state('timesteps')
        ) >> \
    (state['latent']
        - AddNoise(scheduler=pipe.scheduler)
        < state('latent')
        ) >> \
    (state['latent']
        | LatentShow(fps=16, vae=pipe.vae.to('cuda'))
        - LoraOn(loras={
            f"{models}/lora/details.safetensors": 0.8,
            }, pipe=pipe)
        | Loop(state['timesteps'], name="Denoise loop", callback=lambda timestep_index, timestep:
            (Latent(latent=torch.zeros_like(state['latent'].latent)) > state('noise_predict')) >> \
            (state
                | Loop(
                    TileGenerator(
                        Tile(
                            height=36, height_overlap=18,
                            width=64, width_overlap=32,
                            length=48, length_overlap=24,
                            offset=timestep_index*0,
                            ),
                        state['latent']
                    ),
                    name="Tile loop",
                    callback=lambda *tile:
                        state
                            | cnet(
                                timestep_index,
                                timestep,
                                latent=Latent(latent=state['latent'].latent[tile]),
                                image=(state['controlnet_image'].chw().float()/255.0)[:,tile[2],:,scale_slice(tile[3],8),scale_slice(tile[4],8)],
                                controlnet_scale=[1.0],
                            )
                            | unet(timestep_index, timestep, latent=Latent(latent=state['latent'].latent[tile]))
                            > Latent(latent=state['noise_predict'].latent[tile])

                  ) | Debug("Tile loop end")
            ) >> \
            (state
               | Step(pipe.scheduler, timestep, state['noise_predict'], state['latent'])
               - LatentShow(fps=16, vae=pipe.vae.to('cuda'))
               > state("latent")
            )
        )
        | Debug("Denoise loop end")
        | LoraOff(pipe=pipe)
    ) >> \
    (state['latent'] | VaeLatentDecode(vae=pipe.vae.to('cuda')) | VideoShow(fps=16) )
