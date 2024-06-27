import torch
import logging
from latentflow import *
import random

class MergeRegionsLatent(Flow):
    def __init__(self, regions):
        self.regions = regions

    def apply(self, state):
        latent = state['latent'].zeros_like()
        for region in self.regions:
            ((region.latent if not region.shrink['shrink'] else region.latent_shrinked)
                | If(region.shrink is not None and 'shrink' in region.shrink and region.shrink['shrink'],
                    lambda x: x
                    | LatentUnshrink(
                        region.latent,
                        region.mask,
                        padding=0 if region.shrink['shrink_padding'] is None else region.shrink['shrink_padding']//8,
                        )
                    #| Resize(1/region.shrink['shrink_resize'])
                    | state['nn_latent_upscale'](1/region.shrink['shrink_resize'])
                    , desc='unshrink')
                | LatentAdd(latent, mask=(region.mask | Resize(1/region.shrink['shrink_resize'])))
                )

        return latent

class IncrementPixelAuditByMask(Flow):
    def __init__(self, region, tile, mask=None):
        self.region = region
        self.tile = tile
        self.mask = mask

    def apply(self, other):
        tensor = self.region.pixel_infer_count.tensor

        ones = torch.ones_like(tensor[self.tile])
        if self.mask is not None:
            ones = ones * self.mask[self.tile].mask.to(ones.device)

        tensor[self.tile] += ones

        self.region.pixel_infer_count.tensor = tensor

        return other

class CalcBaseMask(Flow):
    def __init__(self, region_index, timestep_index, state):
        self.region_index = region_index
        self.timestep_index = timestep_index
        self.regions = state['regions']
        self.timesteps = state['timesteps']
        self.state = state

    def apply(self, region):

        if self.region_index == 0 and region.mask is None and len(self.state['regions']) > 0:
            mask = Mask(torch.ones_like(self.state['latent'].latent[:,:1,:,:,:]))
            for ri in range(len(self.regions)-1, 0, -1):
                r = self.regions[ri]
                if (r.start_timestep if r.start_timestep is not None else 0) <= self.timestep_index and \
                    self.timestep_index < (r.stop_timestep if r.stop_timestep is not None else len(self.timesteps)):
                    mask = mask - r.mask

            region.mask = mask

        return region

def get_prompt_loras(state, region, tile):
    loras = [region.loras]
    start_frame = state['start_frame'] + tile[2][0]
    start_loras = None
    if region.prompt[start_frame] is None:
        for i in range(start_frame, -1, -1):
            if region.prompt[i] is not None:
                start_loras = region.prompt[i].loras
                break

    loras.append(start_loras)

    for f in tile[2]:
        frame = state['start_frame'] + f
        if region.prompt[frame] is not None and region.prompt[frame].loras is not None:
            loras.append(region.prompt[frame].loras)

    return loras

def get_prompt_cnet(state, region, tile):
    cnets = [region.controlnet_scale]
    start_frame = state['start_frame'] + tile[2][0]
    start_controlnet = None
    if region.prompt[start_frame] is None:
        for i in range(start_frame, -1, -1):
            if region.prompt[i] is not None:
                start_controlnet = region.prompt[i].controlnet
                break

    cnets.append(start_controlnet)

    for f in tile[2]:
        frame = state['start_frame'] + f
        if region.prompt[frame] is not None and region.prompt[frame].controlnet is not None:
            cnets.append(region.prompt[frame].controlnet)

    filtered = [x for x in cnets if x is not None]
    cnet_scale = filtered[-1]
    return cnet_scale


class RegionShedPipeline(Flow):
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

        (state
            | Schedule(scheduler=state['pipe'].scheduler, num_inference_steps=state['num_inference_steps'], strength=state['strength'])
            | Set(state, 'timesteps')
            )

        for region in state['regions']:
            (state
                | Schedule(scheduler=region.scheduler, num_inference_steps=state['num_inference_steps'], strength=state['strength'])
                | Set(region, 'timesteps')
                )

        progress_bar_each_step = state['video_length'] >= 500 or (state['progress_bar_each_step'] is not None  and state['progress_bar_each_step'])

        ((state
            | Loop(state['timesteps'], name="Denoise loop", progress_bar=(not progress_bar_each_step), callback=lambda timestep_index, timestep:
                (state | Loop(state['regions'], name="Region loop", callback=lambda region_index, region:

                    (Value(region_index) | Set(state, 'timestep_index'))  >> \

                    (Noop()
                        | If(region.shrink is None or 'shrink' not in region.shrink or not region.shrink['shrink'],
                        lambda x: state['latent'].clone()
                            | Set(region, 'latent')
                            , 'shrink resize')) >> \

                    (Noop() | If(state['cnet'] is not None, lambda v:
                        region.controlnet_video
                        | If(region.mask is not None and region.shrink is not None and region.shrink['shrink'],
                            lambda v: v
                                | VideoShrink(
                                    region.mask,
                                    padding=0 if region.shrink['shrink_padding'] is None else region.shrink['shrink_padding'],
                                )
                            , desc='shrink')
                        | Set(region, 'controlnet_shrinked_video')
                        , desc='cnet tile')
                    ) >> \

                    (Noop() | If(region.latent_shrinked is None, lambda x:
                        (region.latent
                            | If(region.mask is not None and region.shrink is not None and region.shrink['shrink'],
                                lambda x: x
                                | LatentShrink(
                                    region.mask,
                                    padding=0 if region.shrink['shrink_padding'] is None else region.shrink['shrink_padding']//8,
                                    )
                                , desc='region_mask latent shrink'
                                )
                            | Set(region, 'latent_shrinked')
                        )
                    )) >> \

                    (Latent(torch.zeros_like(region.latent_shrinked.latent).repeat((2 if state['guidance_scale'] > 1.0 else 1),1,1,1,1))
                        | Set(region, 'noise_predict')
                        ) >> \

                    (Tensor(torch.zeros_like(region.latent_shrinked.latent))
                        | Set(region, 'pixel_infer_count'))

                )) >> \

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
                    do_classifier_free_guidance = (state['guidance_scale'] > 1.0),
                    ) | Set(state, 'tile_generator')) >> \

                (state | Loop(
                    state['tile_generator'],
                    name=f"Tile loop {timestep_index+1}/{len(state['timesteps'])} {state['start_frame']}+{state['video_length']}",
                    progress_bar=progress_bar_each_step,
                    callback=lambda tile_index, tile:
                        (state | Loop(state['regions'], name="Region loop", callback=lambda region_index, region:

                            (region | CalcBaseMask(region_index, timestep_index, state)) >> \

                            ((region.latent[tile] if not region.shrink['shrink'] else region.latent_shrinked[:,:,tile[2],:,:])
                                | CFGPrepare(guidance_scale=state['guidance_scale'])
                                | If(state['cnet'] is not None, lambda x: x | state['cnet'](
                                    timestep_index=timestep_index,
                                    timestep=region.timesteps[timestep_index],
                                    image=[x for x in region.controlnet_shrinked_video[:,tile[2],:, :, :].cnet().tensor],
                                    timesteps=region.timesteps,
                                    controlnet_scale=get_prompt_cnet(state, region, tile),
                                    embeddings=region.prompt.embeddings.slice(
                                        tile[2],
                                        do_classifier_free_guidance=state['guidance_scale'] > 1.0),
                                    do_classifier_free_guidance=(state['guidance_scale'] > 1.0),
                                ), desc='cnet')
                                | LoraOn(get_prompt_loras(state, region, tile), pipe=state['pipe'])
                                | state['unet'](
                                    timestep=region.timesteps[timestep_index],
                                    embeddings=region.prompt.embeddings.slice(
                                        tile[2],
                                        do_classifier_free_guidance=state['guidance_scale'] > 1.0),
                                )
                                | LatentAdd(region.noise_predict,
                                    (
                                        slice(0,2 if state['guidance_scale'] > 1.0 else 1),
                                        slice(0,region.latent_shrinked.latent.shape[1]),
                                        tile[2],
                                        slice(0,region.latent_shrinked.latent.shape[3]),
                                        slice(0,region.latent_shrinked.latent.shape[4])
                                    )
                                )
                                | IncrementPixelAuditByMask(region,
                                    (
                                        slice(0,2 if state['guidance_scale'] > 1.0 else 1),
                                        slice(0,region.latent_shrinked.latent.shape[1]),
                                        tile[2],
                                        slice(0,region.latent_shrinked.latent.shape[3]),
                                        slice(0,region.latent_shrinked.latent.shape[4])
                                    )
                                )

                            )
                            | LoraOff(pipe=state['pipe'])
                            | Debug("Region loop end")
                            )
                        )
                )) >> \

                (state | Loop(state['regions'], name="Region loop", callback=lambda region_index, region:
                    (region.noise_predict
                       | Apply(lambda x: NoisePredict(x.latent / region.pixel_infer_count.tensor.repeat((2 if state['guidance_scale'] > 1.0 else 1),1,1,1,1)))
                       #| LatentShow(fps=state['fps'], vae=state['pipe'].vae, vae_batch=state['vae_batch'])
                       | CFGProcess(
                           guidance_scale=state['guidance_scale'],
                           guidance_rescale=state['guidance_rescale'],
                           )
                       | Step(
                           scheduler=region.scheduler,
                           timestep=region.timesteps[timestep_index],
                           latent=(region.latent if not region.shrink['shrink'] else region.latent_shrinked))
                       | Set(region, ('latent' if not region.shrink['shrink'] else 'latent_shrinked'))
                       )
                )) >> \

                (state
                   | MergeRegionsLatent(state['regions'])
                   #| LatentShow(fps=state['fps'], vae=state['pipe'].vae, vae_batch=state['vae_batch'])
                   | Set(state, 'latent')
                )
            )
            | Debug("Denoise loop end")
        )
        | Save(path=f'{self.samples_dir}/%datetime%/latent.pth')
        | Save(path=f'{self.samples_dir}/last/latent.pth')
        )

        return state
