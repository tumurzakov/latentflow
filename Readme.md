# LatentFlow is pseudo functional stable diffusion

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
from latentflow.debug import Debug
from latentflow.invert import Invert
from latentflow.bypass import Bypass

import importlib
import latentflow.video_load
importlib.reload(latentflow.video_load)
from latentflow.video_load import VideoLoad


unet = Unet(
    unet=pipe.unet.to('cuda'),
    scheduler=pipe.scheduler,
    guidance_scale=10,
)

state = State()
video = \
    (Video(video=torch.randn((16,288,512,3))) \
        | Debug("Video to video")
        | VideoLoad('src/tests_func/bunny.mp4', device='cuda', video_length=48) \
        | VideoShow(fps=16) \
        | VaeVideoEncode(vae=pipe.vae.to('cuda')) \
        | Invert(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder, unet=pipe.unet, scheduler=pipe.scheduler) \
        | LatentShow(fps=16, vae=pipe.vae.to('cuda')) \
        > state("latent") \
        ) >> \
    (Latent(latent=torch.randn((1,4,16,36,64)), device='cuda') \
        - Debug("Text to video")
        - Noise(scheduler=pipe.scheduler) \
        - LatentShow(fps=16, vae=pipe.vae.to('cuda')) \
        < state("latent") \
        ) >> \
    (Prompt("a cat walking down city street") \
        | CompelPromptEncode(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder.to('cuda')) \
        > state("embeddings") \
        ) >> \
    (state \
        | Schedule(scheduler=pipe.scheduler, num_inference_steps=10, strength=0.7) \
        > state('timesteps') \
        ) >> \
    (state['latent'] \
        - Debug("Adding noise")
        - AddNoise(scheduler=pipe.scheduler) \
        < state('latent') \
        ) >> \
    (state
        | Debug("Diffusion loop")
        | Diffuse(callback=lambda timestep, state: state | unet)
        | VaeLatentDecode(vae=pipe.vae.to('cuda'))
    ) \
    | VideoShow(fps=16)
```
