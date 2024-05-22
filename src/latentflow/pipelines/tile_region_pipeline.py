import torch
import logging
from latentflow import *
import random

class LorasOn(Flow):
    def __init__(self, loras, pipe, frames):
        self.loras = loras
        self.pipe = pipe
        self.frames = frames

    def apply(self, other):

        frames = list(self.loras.keys())
        frames.sort()
        for i, frame in enumerate(frames):
            start = (0 if i == 0 else frames[i-1])
            stop = frame

            if (start <= self.frames[0] < frame) and (start <= self.frames[15] < frame):
                (Noop() | LoraOn(self.loras[frame], pipe=self.pipe))

        return other

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
                , desc='strength')
            | Loop(state['timesteps'], name="Denoise loop", progress_bar=state['video_length'] < 500, callback=lambda timestep_index, timestep:
                (Latent(torch.zeros_like(state['latent'].latent).repeat((2 if state['guidance_scale'] > 1.0 else 1),1,1,1,1)) > state('noise_predict')) >> \

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
                    do_classifier_free_guidance = (state['guidance_scale'] > 1.0),
                    ) | Set(state, 'tile_generator')) >> \

                (state | Loop(
                    state['tile_generator'],
                    name=f"Tile loop {timestep_index+1}/{len(state['timesteps'])}",
                    progress_bar=state['video_length'] >= 500,
                    callback=lambda tile_index, tile:
                        (state['latent'][tile] | Set(state, "tile_latent")) >> \

                        (state | If(state['cnet'] is not None, lambda x:
                            (state['controlnet_video']
                                | Apply(lambda x: x[:,tile[2],slice_scale(tile[3],8),slice_scale(tile[4],8),:])
                                #| VideoShow(fps=state['fps'])
                                | Set(state, "tile_controlnet_video")
                            )
                        , desc='cnet')) >> \

                        (state | Loop(state['regions'], name="Region loop", callback=lambda region_index, region:

                            (region | If(region.mask is None or region.mask[tile].mask.sum() > 0, lambda x:

                                (Noop() | If(state['cnet'] is not None, lambda v:
                                    state['tile_controlnet_video']
                                    | If(region.mask is not None and state['shrink'] is not None and state['shrink'],
                                        lambda v: v
                                            | VideoShrink(
                                                region.mask[tile].cfg(guidance_scale=state['guidance_scale']),
                                                padding=0 if state['shrink_padding'] is None else state['shrink_padding'],
                                            )
                                            | If(region_index > 0 and state['shrink_resize'] is not None,
                                                lambda x: x | Resize(state['shrink_resize']), desc='shrink resize')
                                        , desc='region.mask')
                                    | Set(state, 'tile_controlnet_shrinked_video')
                                    , desc='cnet tile')
                                ) >> \

                                (state['tile_latent']
                                    | If( \
                                        (region.start_timestep if region.start_timestep is not None else 0) <= timestep_index and \
                                        timestep_index < ( \
                                            region.stop_timestep if region.stop_timestep is not None \
                                            else len(state['timesteps'])
                                        ) and \
                                        state['tile_latent'].latent.shape[2] > 0 and \
                                        state['tile_latent'].latent.shape[3] > 0 and \
                                        state['tile_latent'].latent.shape[4] > 0 and \
                                        ( \
                                            state['tile_controlnet_shrinked_video'] is None or \
                                            ( \
                                                state['tile_controlnet_shrinked_video'].chw().shape[1] > 0 and \
                                                state['tile_controlnet_shrinked_video'].chw().shape[3] > 0 and \
                                                state['tile_controlnet_shrinked_video'].chw().shape[4] > 0
                                            ) \
                                        )
                                        ,
                                        lambda x: x
                                            | CFGPrepare(guidance_scale=state['guidance_scale'])
                                            | Set(state, 'tile_latent_cfg')
                                            | If(region.mask is not None and state['shrink'] is not None and state['shrink'],
                                                lambda x: x
                                                    | LatentShrink(
                                                        region.mask[tile].cfg(guidance_scale=state['guidance_scale']),
                                                        padding=0 if state['shrink_padding'] is None else state['shrink_padding']//8,
                                                    )
                                                    | If(region_index > 0 and state['shrink_resize'] is not None,
                                                        lambda x: x | Resize(state['shrink_resize']), desc='shrink resize')
                                                , desc='region_mask')
                                            | If(state['cnet'] is not None, lambda x: x | state['cnet'](
                                                timestep_index=timestep_index,
                                                timestep=state['timesteps'][timestep_index],
                                                image=[x for x in state["tile_controlnet_shrinked_video"].cnet().tensor],
                                                timesteps=state['timesteps'],
                                                controlnet_scale=get_prompt_cnet(state, region, tile),
                                                #embeddings=region.prompt.embeddings,
                                                embeddings=region.prompt.embeddings.slice(
                                                    tile[2],
                                                    do_classifier_free_guidance=state['guidance_scale'] > 1.0),
                                                do_classifier_free_guidance=(state['guidance_scale'] > 1.0),
                                            ), desc='cnet')
                                            | LoraOn(get_prompt_loras(state, region, tile), pipe=state['pipe'])
                                            | state['unet'](
                                                timestep=state['timesteps'][timestep_index],
                                                #embeddings=region.prompt.embeddings,
                                                embeddings=region.prompt.embeddings.slice(
                                                    tile[2],
                                                    do_classifier_free_guidance=state['guidance_scale'] > 1.0),
                                            )
                                            | If(region.mask is not None and state['shrink'] is not None and state['shrink'],
                                                lambda x: x
                                                    | If(region_index > 0  and state['shrink_resize'] is not None,
                                                        lambda x: x | Resize(1/state['shrink_resize']), desc='shrink resize')
                                                    | LatentUnshrink(
                                                        state['tile_latent_cfg'],
                                                        region.mask[tile].cfg(guidance_scale=state['guidance_scale']),
                                                        padding=0 if state['shrink_padding'] is None else state['shrink_padding']//8,
                                                    )
                                                , desc='region.mask')
                                            | If(region.mask is not None,
                                                lambda x: x | region.mask[tile].cfg(guidance_scale=state['guidance_scale']), desc='region.mask')

                                    , desc=f"Region {region_index} {state['tile_latent']}")
                                )
                                | LatentAdd(state['noise_predict'], tile)
                                | IncrementPixelAuditByMask(state, tile, region.mask)
                                | LoraOff(pipe=state['pipe'])
                                | Debug("Region loop end")
                            , 'region mask is not empty'))
                        ))
                )) >> \

                (state['noise_predict']
                   | Apply(lambda x: NoisePredict(x.latent / state['pixel_infer_count'].tensor.repeat((2 if state['guidance_scale'] > 1.0 else 1),1,1,1,1)))
                   | CFGProcess(
                       guidance_scale=state['guidance_scale'],
                       guidance_rescale=state['guidance_rescale'],
                       )
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
        )

        return state
