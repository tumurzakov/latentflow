# LatentFlow is a functional-style API for Stable Diffusion

* 2024-01-11 LoRA and ControlNet added
* 2023-01-10 Initial release

```python

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

import torch

unet = Unet(
    unet=pipe.unet.to('cuda'),
    scheduler=pipe.scheduler,
    guidance_scale=10,
)

cnet = ControlNet(
    controlnet=pipe.controlnet.to('cuda'),
    scheduler=pipe.scheduler,
)

state = State()
video = \
    (Debug("Video to video")
        | VideoLoad(f'input/in.mp4', device='cuda', video_length=48)
        - VideoShow(fps=16)
        | VaeVideoEncode(
            vae=pipe.vae.to('cuda'),
            cache=f'infer/latents.pth',
            video_length=48,
            )
        - LatentShow(fps=16, vae=pipe.vae.to('cuda'))
        | Invert(
            tokenizer=pipe.tokenizer,
            text_encoder=pipe.text_encoder,
            unet=pipe.unet,
            scheduler=pipe.scheduler,
            cache=f'infer/inv_latents.pth',
            )
        - LatentShow(fps=16, vae=pipe.vae.to('cuda'))
        | DebugHash(lambda latent: latent.latent)
        > state("latent")
        ) >> \
    (Latent(latent=torch.randn((1,4,48,36,64)), device='cuda')
        - Debug("Text to video")
        - Noise(scheduler=pipe.scheduler)
        - LatentShow(fps=16, vae=pipe.vae.to('cuda'))
        < state("latent")
        ) >> \
    (Prompt(prompt=
               "(a cat walking down city street)1.0,"
            negative_prompt=
               "deformed, distorted, disfigured)1.0,"
               "poorly drawn, bad anatomy, wrong anatomy,"
               "extra limb,missing limb, floating limbs,"
               "(mutated hands and fingers)1.0,"
               "disconnected limbs, mutation, mutated,"
               "ugly, disgusting, blurry, amputation,"
           )
        | CompelPromptEncode(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder.to('cuda'))
        | Apply(lambda embeddings: PromptEmbeddings(embeddings=embeddings.embeddings.repeat(48, 1, 1)))
        | DebugHash(lambda embeddings: embeddings.embeddings)
        > state("embeddings")
        ) >> \
    (Debug("Loading controlnet")
      | VideoLoad([
              f'lineart_out/lineart.mp4',
          ], device='cuda', video_length=48, width=512, height=288)
      | VideoShow(fps=16)
      > cnet
      ) >> \
    (state \
        | Schedule(scheduler=pipe.scheduler, num_inference_steps=10)
        > state('timesteps')
        ) >> \
    (state['latent']
        - AddNoise(scheduler=pipe.scheduler)
        < state('latent')
        ) >> \
    (state
        | LoraOn(loras={
            f"lora/details.safetensors": 0.8,
            }, pipe=pipe)
        | Diffuse(callback=lambda timestep, state:
                  state
                  | cnet(controlnet_scale=[1.0])
                  | unet
                 )
        | LoraOff(pipe=pipe)
        | VaeLatentDecode(vae=pipe.vae.to('cuda'))
    ) \
    | VideoShow(fps=16)
```

* `|`  is pipe (like unix pipe)
* `>` write value in state (like in unix too)
* `-` bypass for pipe
* `<` bypass for write
* `>>` go to next statement 
