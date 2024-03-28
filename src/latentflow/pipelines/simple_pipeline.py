import torch
import logging
from latentflow import *
import random

class SimplePipeline(Flow):
    """
    Requirements in state:
        latent                Required
        prompt                Required
        pipe                  Required
        unet                  Required

        num_inference_steps   Required
        guidance_scale        Required
        strength              Optional

        video_length          Required
        width                 Required
        height                Required
        fps                   Required
        vae_batch             Optional
        start_frame           Optional

        seed                  Optional
    """

    def __init__(self, infer_dir: str, samples_dir: str = None):
        self.infer_dir = infer_dir
        self.samples_dir = samples_dir
        if samples_dir is None:
            samples_dir = f'{infer_dir}/samples'

    def apply(self, state):
        (Seed(state['seed']) | Set(state, 'seed'))

        (state \
            | Schedule(scheduler=state['pipe'].scheduler, num_inference_steps=state['num_inference_steps'], strength=state['strength'])
            | Set(state, 'timesteps')
            )

        ((state['latent']
            | If(state['strength'] is not None, lambda x: x
                | AddNoise(scheduler=state['pipe'].scheduler, timesteps=state['timesteps'])
                )

            - LatentShow(fps=state['fps'], vae=state['pipe'].vae)
            | Loop(state['timesteps'], name="Denoise loop", progress_bar=True, callback=lambda timestep_index, timestep:
                (state['latent']
                    | CFGPrepare(guidance_scale=state['guidance_scale'])
                    | state['unet'](
                        timestep=timestep,
                        embeddings=state['prompt'].embeddings,
                    )
                    | CFGProcess(guidance_scale=state['guidance_scale'])
                    | Step(
                        scheduler=state['pipe'].scheduler,
                        timestep=timestep,
                        latent=state['latent'])
                    | Set(state, 'latent')
                )
            )
            | Debug("Denoise loop end")
        )
        | Save(path=f'{self.samples_dir}/%datetime%/latent.pth')
        | Save(path=f'{self.samples_dir}/last/latent.pth')
        - LatentShow(fps=state['fps'], vae=state['pipe'].vae, vae_batch=state['vae_batch'])
        | VaeLatentDecode(vae=state['pipe'].vae, vae_batch=state['vae_batch'])
        | Save(path=f'{self.samples_dir}/%datetime%/video.mp4', fps=state['fps'])
        | Save(path=f'{self.samples_dir}/last/video.mp4', fps=state['fps'])
        | VideoShow(fps=state['fps'])
        | Set(state, "video")
        )

        return state
