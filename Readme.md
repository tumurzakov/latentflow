# LatentFlow is a functional-style API for Stable Diffusion

## Changelog

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

import torch
import latentflow as lf

pipe = lf.AnimateDiffPipeline.load(
    pretrained_model_path=f'{models}/stable_diffusion-v1-5',
    motion_module_path=f'{models}/mm_sd_v15_v2.ckpt',
    motion_module_config_path=f'{models}/inference.yaml',
    scheduler_class_name='EulerDiscreteScheduler',
)

state = lf.State({
    'width': 512,
    'height': 512,
    'video_length': 16,
    'num_inference_steps': 10,
})

(video = \
    (lf.Latent(shape=(1,4,state['video_length'],state['width']//8,state['height']//8), device='cuda')
        | lf.Noise(scheduler=pipe.scheduler)
        | lf.LatentShow(fps=16, vae=pipe.vae.to('cuda'))
        > state("latent")
        ) >> \

    (state['latent'] \
        | lf.Pipeline(
            pipe=pipe,
            num_inference_steps=state['num_inference_steps'],
            prompt='a cat walking down city street',
        )
        | lf.VaeLatentDecode(vae=pipe.vae)
        | lf.VideoShow(fps=16)
        )
)
```
