from einops import rearrange

from latentflow.state import State
from latentflow.video import Video
from latentflow.latent import Latent, LatentAdd, NoisePredict
from latentflow.tile import Tile
from latentflow.prompt import Prompt
from latentflow.vae_video_encode import VaeVideoEncode
from latentflow.vae_latent_decode import VaeLatentDecode
from latentflow.unet import Unet, CFGPrepare, CFGProcess
from latentflow.prompt_embeddings import PromptEmbeddings
from latentflow.a1111_prompt_encode import A1111PromptEncode
from latentflow.compel_prompt_encode import CompelPromptEncode
from latentflow.video_load import VideoLoad
from latentflow.video_show import VideoShow
from latentflow.latent_show import LatentShow
from latentflow.noise import Noise
from latentflow.add_noise import AddNoise
from latentflow.schedule import Schedule
from latentflow.debug import Debug, DebugHash, Info
from latentflow.invert import Invert
from latentflow.bypass import Bypass
from latentflow.lora import LoraOn, LoraOff
from latentflow.apply import Apply
from latentflow.controlnet import ControlNet
from latentflow.loop import Loop
from latentflow.tile import Tile, TileGenerator, UniformFrameTileGenerator
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
from latentflow.ip_adapter_prompt_encode import IPAdapterPromptEncode
from latentflow.comfy_ip_adapter_prompt_encode import ComfyIPAdapterPromptEncode
from latentflow.mandelbrot_noise import MandelbrotNoise
from latentflow.save import Save, Load
from latentflow.video_rembg import VideoRembg
from latentflow.video_face_crop import VideoFaceCrop

import torch
torch.backends.cuda.matmul.allow_tf32 = True

unet = Unet(
    unet=pipe.unet.to('cuda'),
    scheduler=pipe.scheduler,
    fp16=True,
)

cnet = ControlNet(
    controlnet=pipe.controlnet.to('cuda', dtype=torch.float16),
    scheduler=pipe.scheduler,
)

comfy_ip_adapter_face_plus = ComfyIPAdapterPromptEncode(
    ip_adapter_path=f'{models}/IP-Adapter/models/ip-adapter-plus-face_sd15.bin',
    clip_path=f'{models}/IP-Adapter/models/image_encoder/pytorch_model.bin',
    pipe = pipe,
    noise = 0.0,
    weight = 1.0,
)

comfy_ip_adapter_faceid = ComfyIPAdapterPromptEncode(
    ip_adapter_path=f'{models}/IP-Adapter/models/ip-adapter-faceid-plusv2_sd15.bin',
    clip_path=f'{models}/IP-Adapter/models/image_encoder/pytorch_model.bin',
    pipe = pipe,
    noise = 0.0,
    weight = 1.0,
)

state = State({
    'width': 512,
    'height': 288,
    'video_length': 48,
    'vae_batch': 12,
    'num_inference_steps': 30,
    'guidance_scale': 7,
    'start_frame': 0,
    'fps': 16,
})

state.update({
    'tile': {
        'length':48,
        'width':512,
        'height':288,
        'length_overlap': 32,
        'height_overlap': 256,
        'width_overlap': 256,
        'offset': 0,

        'strides': 3,
        'overlap': 8,
    },
})

video = (
    (Debug("Video to video")
        | VideoLoad(
            f'{data}/input/in.mp4', 
            device='cuda', 
            video_length=state['video_length'], 
            start_frame=state['start_frame'], 
            width=state['width'],
            height=state['height'],
        )
        | Set(state, 'input_video')
        - VideoShow(fps=state['fps'])
        | VaeVideoEncode(
            vae=pipe.vae.to('cuda'),
            cache=f'{data}/infer/latents.pth',
            video_length=state['video_length'],
            )
        - LatentShow(fps=state['fps'], vae=pipe.vae.to('cuda'))
        | Invert(
            tokenizer=pipe.tokenizer,
            text_encoder=pipe.text_encoder,
            unet=pipe.unet,
            scheduler=pipe.scheduler,
            cache=f'{data}/infer/inv_latents.pth',
            temporal_context=state['tile']['length'],
            )
        - LatentShow(fps=state['fps'], vae=vae, vae_batch=state['vae_batch'])
        > state("latent")
        ) >> \
    (Debug("Load facecrop video 1")
        | VideoLoad(
            f'{data}/in_720.mp4', 
            device='cuda', 
            video_length=state['video_length'], 
            start_frame=state['start_frame'],
        )
        | VideoFaceCrop(cache=f'{data}/infer/facecrop1.pth', padding_percent=0.1, size=(224,224))
        | VideoShow(fps=state['fps'])
        > state('face_video_plus')
        ) >> \
    (Debug("Load facecrop video 2")
        | VideoLoad(
            f'{data}/in_720.mp4', 
            device='cuda', 
            video_length=state['video_length'], 
            start_frame=state['start_frame'],
        )
        | VideoFaceCrop(cache=f'{data}/infer/facecrop2.pth', padding_percent=0.4, size=(224,224))
        | VideoShow(fps=state['fps'])
        > state('face_video_faceid')
        ) >> \
    (Latent(shape=(1,4,state['video_length'],state['height']//8,state['width']//8), device='cuda')
        | Noise(scheduler=pipe.scheduler, device='cuda')
        - LatentShow(fps=state['fps'], vae=vae, vae_batch=state['vae_batch'])
        < state("latent")
        ) >> \
    (Prompt(
            prompt=
               "(young woman in haute couture dress)1.0,"
               "boring office on background, working employees, computers, phones, copiers, working people,"
               "cyberpunk, fantasy, sci-fi, stunning, concept, artstation, acid colors,"
               "octane render, Unreal engine, cg unity, hdr, wallpaper, neonpunkai, lineart,",
            negative_prompt=
               "deformed, distorted, disfigured)1.0,"
               "poorly drawn, bad anatomy, wrong anatomy,"
               "extra limb,missing limb, floating limbs,"
               "(mutated hands and fingers)1.0,"
               "disconnected limbs, mutation, mutated,"
               "ugly, disgusting, blurry, amputation,",
            frames=list(range(0,state['video_length'])),
            #embeddings=PromptEmbeddings()|Load(f'{data}/infer/embeddings_{state['video_length']}.pth')
        )
        | CompelPromptEncode(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder)
        | comfy_ip_adapter_faceid(state['face_video_faceid'])
        | comfy_ip_adapter_face_plus(state['face_video_plus'])
        | Save(f'{data}/infer/embeddings_48.pth')
        > state("prompt")
        ) >> \
    (Debug("Loading controlnet")
        | VideoLoad(
            [
                f'{data}/lineart_out/lineart.mp4',
            ], 
            device='cuda', 
            video_length=state['video_length'], 
            width=state['width'], 
            height=state['height'], 
            start_frame=state['start_frame'],
        )
        | VideoShow(fps=state['fps'])
        > state('controlnet_image')
        ) >> \
    (state \
        | Schedule(scheduler=pipe.scheduler, num_inference_steps=state['num_inference_steps'])
        > state('timesteps')
        ) >> \
    (state['latent']
        - LatentShow(fps=state['fps'], vae=vae)
        | LoraOn(loras={
            # f"{models}/lora/details.safetensors": 0.8,
            # f"{models}/lora/phblue.safetensors": 0.5,
            # f"{models}/lora/Neonpunkai.safetensors": 0.8,
            # f"{models}/lora/Cyberpunk_fantasy.safetensors": 0.3,
            f"{models}/IP-Adapter/models/lora/ip-adapter-faceid-plusv2_sd15_lora.safetensors": 0.6,
            }, pipe=pipe)
        | Loop(state['timesteps'], name="Denoise loop", progress_bar=True, callback=lambda timestep_index, timestep:
            (Latent(torch.zeros_like(state['latent'].latent).repeat(2,1,1,1,1)) > state('noise_predict')) >> \
            (Tensor(torch.zeros_like(state['latent'].latent)) > state('pixel_infer_count')) >> \
            (state
                 | Loop(
                    UniformFrameTileGenerator(
                        timestep_index=timestep_index,
                        context_size=state['tile']['length'],
                        strides=state['tile']['strides'],
                        overlap=state['tile']['overlap'],
                        latent=state['latent'],
                        pixel_infer_count=state['pixel_infer_count'],
                    ),
                    name="Tile loop",
                    callback=lambda tile_index, tile:
                        (state['latent'][tile] > state("tile_latent")) >> \

                        (state['controlnet_image'][:,tile[2],slice_scale(tile[3],8),slice_scale(tile[4],8),:].cnet() > state("tile_controlnet_image")) >> \
                     
                        state['tile_latent']
                            | CFGPrepare(guidance_scale=state['guidance_scale'])
                            | cnet(
                                timestep_index=timestep_index,
                                timestep=timestep,
                                image=state["tile_controlnet_image"].tensor,
                                timesteps=state['timesteps'],
                                embeddings=state['prompt'].embeddings.slice(slice_scale(tile[2],2)),
                                controlnet_scale=[1.0],
                            )
                            | unet(
                                timestep=timestep,
                                embeddings=state['prompt'].embeddings.slice(slice_scale(tile[2],2)),
                            )
                            | LatentAdd(state['noise_predict'], tile)


                  ) | Debug("Tile loop end")
            ) >> \
            (state['noise_predict']
               | Apply(lambda x: NoisePredict(x.latent / state['pixel_infer_count'].tensor.repeat(2,1,1,1,1)))
               | CFGProcess(guidance_scale=state['guidance_scale'])
               | Step(
                   scheduler=pipe.scheduler,
                   timestep=timestep,
                   latent=state['latent'])
               - LatentShow(fps=16, vae=vae, vae_batch=state['vae_batch'])
               > state("latent")
            )
        )
        | Debug("Denoise loop end")
    )
    | Save(path=f'{data}/samples/%datetime%/latent.pth')
    | LatentShow(fps=16, vae=vae, vae_batch=state['vae_batch'])
)
