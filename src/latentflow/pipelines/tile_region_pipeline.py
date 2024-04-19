import torch
import logging
from latentflow import *
import random

class IncrementPixelAuditByMask(Flow):
    def __init__(self, state, tile, mask):
        self.state = state
        self.tile = tile
        self.mask = mask

    def apply(self, other):
        tensor = self.state['pixel_infer_count'].tensor

        ones = torch.ones_like(tensor[self.tile])
        if self.mask is not None:
            ones = ones * self.mask[self.tile].mask.to(ones.device)

        tensor[self.tile] += ones
        self.state['pixel_infer_count'].tensor = tensor
        return other

class CalcBaseMask(Flow):
    def __init__(self, region_index, timestep_index, state):
        self.region_index = region_index
        self.timestep_index = timestep_index
        self.regions = state['regions']
        self.timesteps = state['timesteps']

    def apply(self, region):


        if self.region_index == 0 and region.mask is not None:
            mask = Mask(torch.ones_like(region.mask.mask))
            for ri in range(len(self.regions)-1, 0, -1):
                r = self.regions[ri]
                if (r.start_timestep if r.start_timestep is not None else 0) <= self.timestep_index and \
                    self.timestep_index < (r.stop_timestep if r.stop_timestep is not None else len(self.timesteps)):
                    mask = mask - r.mask

            region.mask = mask

        return region

class TileRegionPipeline(Flow):
    """
    Requirements in state:
        latent                Required
        controlnet_video      Optional
        regions               Required

        pipe                  Required
        unet                  Required
        cnet                  Optional

        num_inference_steps   Required
        guidance_scale        Required
        strength              Optional

        video_length          Required
        width                 Required
        height                Required
        fps                   Required
        vae_batch             Optional
        start_frame           Optional

        tile.length           Required
        tile.width            Required
        tile.height           Required
        tile.length_overlap   Required
        tile.width_overlap    Required
        tile.height_overlap   Required

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
            | Loop(state['timesteps'], name="Denoise loop", progress_bar=True, callback=lambda timestep_index, timestep:
                (Latent(torch.zeros_like(state['latent'].latent).repeat(2,1,1,1,1)) > state('noise_predict')) >> \

                (Tensor(torch.zeros_like(state['latent'].latent)) > state('pixel_infer_count')) >> \

                (TileGenerator(
                    Tile(
                        height=state['tile']['height']//8,
                        width=state['tile']['width']//8,
                        length=state['tile']['length'],
                        length_overlap=state['tile']['length_overlap'],
                        height_overlap=state['tile']['height_overlap']//8,
                        width_overlap=state['tile']['width_overlap']//8,
                        length_offset=timestep_index if state['tile']['length'] < state['video_length'] else 0,
                        height_offset=random.randint(0, state['tile']['height_overlap']//8) if state['tile']['height'] < state['height'] else 0,
                        width_offset=random.randint(0, state['tile']['width_overlap']//8) if state['tile']['width'] < state['width'] else 0,
                        ),
                    state['latent'],
                    do_classifier_free_guidance=True,
                    #pixel_infer_count=state['pixel_infer_count'],
                    ) | Set(state, 'tile_generator')) >> \

                (state | Loop(
                    state['tile_generator'],
                    name="Tile loop",
                    callback=lambda tile_index, tile:
                        (state['latent'][tile] | Set(state, "tile_latent")) >> \

                        (state | If(state['cnet'] is not None, lambda x:
                            (state['controlnet_video']
                                | Apply(lambda x: x[:,tile[2],slice_scale(tile[3],8),slice_scale(tile[4],8),:])
                                - VideoShow(fps=state['fps'])
                                | Set(state, "tile_controlnet_video")
                            )
                        )) >> \

                        (state | Loop(state['regions'], name="Region loop", callback=lambda region_index, region:

                            (region | CalcBaseMask(region_index, timestep_index, state)) >> \

                            (state['tile_latent']
                                | If(
                                    (region.start_timestep if region.start_timestep is not None else 0) <= timestep_index and \
                                    timestep_index < (region.stop_timestep if region.stop_timestep is not None else len(state['timesteps'])),
                                    lambda x: x
                                        | CFGPrepare(guidance_scale=state['guidance_scale'])
                                        | Set(state, 'tile_latent_cfg')
                                        | If(region.mask is not None, lambda x: x | LatentShrink(region.mask[tile].cfg(guidance_scale=state['guidance_scale']), padding=2))
                                        #| If(region.scale is not None, lambda x: x | Resize(scale_factor=region.scale))
                                        | If(state['cnet'] is not None, lambda x: x | state['cnet'](
                                            timestep_index=timestep_index,
                                            timestep=timestep,
                                            image=[x for x in (state["tile_controlnet_video"]
                                                | If(region.mask is not None, lambda v: v | VideoShrink(region.mask[tile].cfg(guidance_scale=state['guidance_scale']), padding=16))
                                                #| If(region.scale is not None, lambda v: v | Resize(scale_factor=region.scale))
                                                ).cnet().tensor],
                                            timesteps=state['timesteps'],
                                            controlnet_scale=region.controlnet_scale,
                                            embeddings=region.prompt.embeddings.slice(tile[2]),
                                            #embeddings=region.prompt.embeddings.slice(tile[2]) | AddFrameEncoding(
                                            #    frame=tile[2][0],
                                            #    tokenizer=state['pipe'].tokenizer,
                                            #    text_encoder=state['pipe'].text_encoder,
                                            #),
                                            #embeddings=region.prompt.embeddings.slice(tile[2]) | AddTileEncoding(
                                            #    tile=tile,
                                            #    tokenizer=state['pipe'].tokenizer,
                                            #    text_encoder=state['pipe'].text_encoder),

                                        ))
                                        | LoraOn(region.loras, pipe=state['pipe'])
                                        | state['unet'](
                                            timestep=timestep,
                                            embeddings=region.prompt.embeddings.slice(tile[2]),
                                            #embeddings=region.prompt.embeddings.slice(tile[2]) | AddFrameEncoding(
                                            #    frame=tile[2][0],
                                            #    tokenizer=state['pipe'].tokenizer,
                                            #    text_encoder=state['pipe'].text_encoder,
                                            #),
                                            #embeddings=region.prompt.embeddings.slice(tile[2]) | AddTileEncoding(
                                            #    tile=tile,
                                            #    tokenizer=state['pipe'].tokenizer,
                                            #    text_encoder=state['pipe'].text_encoder),

                                        )
                                        #| If(region.scale is not None, lambda x: x | Resize(scale_factor=1/region.scale))
                                        | If(region.mask is not None, lambda x: x | LatentUnshrink(state['tile_latent_cfg'], region.mask[tile].cfg(guidance_scale=state['guidance_scale']), padding=2))
                                        | If(region.mask is not None, lambda x: x | region.mask[tile].cfg(guidance_scale=state['guidance_scale']))
                                        | LatentAdd(state['noise_predict'], tile)
                                        | IncrementPixelAuditByMask(state, tile, region.mask)
                                        | LoraOff(pipe=state['pipe'])
                                )
                            )
                            | Debug("Region loop end")
                        ))

                )) >> \

                (state['noise_predict']
                   | Apply(lambda x: NoisePredict(x.latent / state['pixel_infer_count'].tensor.repeat(2,1,1,1,1)))
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
        #| LatentShow(fps=state['fps'], vae=state['pipe'].vae, vae_batch=state['vae_batch'])
        #| VaeLatentDecode(vae=state['pipe'].vae, vae_batch=state['vae_batch'])
        #| Save(path=f'{self.samples_dir}/%datetime%/video.mp4', fps=state['fps'])
        #| Save(path=f'{self.samples_dir}/last/video.mp4', fps=state['fps'])
        #| VideoShow(fps=state['fps'])
        | Set(state, "video")
        )

        return state
