# LatentFlow is a functional-style API for Stable Diffusion

## Changelog

* 2024-02-16 ComfyUI Any node support added
* 2024-02-07 Constant improvements
* 2024-01-15 Region prompts added
* 2024-01-11 Simple tile processing added
* 2024-01-11 LoRA and ControlNet added
* 2023-01-10 Initial release


## Description

Let's look at diffusion generation from different angle.
It is just some function applied to latent tensor.

```
Video -> Latent | Diffusion -> Video
```

This framework mostly for video generation.

## Examples

### Simpliest example

```python

import latentflow as lf

pipe = lf.AnimateDiffPipeline.load(
    pretrained_model_path=f'{models}/stable-diffusion-v1-5',
    motion_module_path=f'{models}/mm_sd_v15_v2.ckpt',
    motion_module_config_path=f'{models}/mm_sd_v15_v2.yaml',
    scheduler_class_name='EulerDiscreteScheduler',
    fp16=False,
    xformers=False,
)

state = lf.State({
    'width': 512,
    'height': 288,
    'video_length': 16,
    'num_inference_steps': 20,
    'prompt': 'a cat walking down city street',
    'fps': 16,
})

video = (
    (lf.Latent()
        | pipe(
            num_inference_steps=state['num_inference_steps'],
            prompt=state['prompt'],
            video_length=state['video_length'],
            temporal_context=state['video_length'],
            width=state['width'],
            height=state['height'],
        )
        | lf.VaeLatentDecode(vae=pipe.vae, vae_batch=4)
        | lf.VideoShow(fps=state['fps'])
        )
)
```

### ComfyUI Node example
```python
import sys

sys.path.insert(0, f'/path/to/latentflow/src')
sys.path.insert(0, f'/path/to/ComfyUI')

import latentflow as lf
import nodes
import torch

state = lf.State({})

(lf.ComfyNode(nodes.CheckpointLoaderSimple, ckpt_name='revanimated.safetensors').apply()
    | lf.Set(state, "checkpoint")
    | lf.ComfyNode(nodes.EmptyLatentImage, width=512, height=512, batch_size=1)
    | lf.Error("Test")
    | lf.ComfyNode(
        nodes.KSampler,
        model=state['checkpoint']['MODEL'],
        positive=lf.ComfyNode(nodes.CLIPTextEncode, text="a cat", clip=state["checkpoint"]["CLIP"]).apply(),
        negative=lf.ComfyNode(nodes.CLIPTextEncode, text="", clip=state["checkpoint"]["CLIP"]).apply(),
    )
    | lf.ComfyNode(nodes.VAEDecode, vae=state['checkpoint']["VAE"])
    | lf.Apply(lambda x: lf.Video('HWC', torch.stack([x['IMAGE']])))
    | lf.VideoShow()
)
```
