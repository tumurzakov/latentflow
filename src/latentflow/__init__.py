import logging

try:
    from latentflow.flow import Flow
    from latentflow.state import State
    from latentflow.state import Context
    from latentflow.seed import Seed
    from latentflow.video import Video, VideoAdd
    from latentflow.latent import Latent, LatentAdd, LatentSet, NoisePredict
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
    from latentflow.debug import Debug, DebugHash, Info, Error, DebugCUDAUsage
    from latentflow.invert import Invert
    from latentflow.bypass import Bypass
    from latentflow.lora import LoraOn, LoraOff, LoraMerge, LoraInitTrain
    from latentflow.apply import Apply, Call
    from latentflow.controlnet import ControlNet,ControlNetLatent
    from latentflow.loop import Loop
    from latentflow.tile import Tile, TileGenerator, UniformFrameTileGenerator, AddTileEncoding
    from latentflow.step import Step
    from latentflow.tensor import Tensor,TensorAdd
    from latentflow.mask import MaskEncode, Mask, LatentMaskCut, VideoMaskCut, LatentMaskMerge, LatentMaskCrop, MaskShow, MaskFlatten
    from latentflow.region import Region
    from latentflow.noop import Noop
    from latentflow.interpolate import Interpolate
    from latentflow.flow import If, Set, Switch
    from latentflow.nn_latent_upscale import NNLatentUpscale
    from latentflow.slice_utils import slice_scale
    from latentflow.mandelbrot_noise import MandelbrotNoise
    from latentflow.save import Save, Load
    from latentflow.image import LoadImage
    from latentflow.latent_interpolate import LatentInterpolate
    from latentflow.adain import Adain
    from latentflow.pipeline import Pipeline
    from latentflow.sd_upscale import SDUpscale
    from latentflow.res_adapter import ResAdapter
except Exception as e:
    logging.error("Error %s", str(e))


try:
    from latentflow.animatediff_pipeline import AnimateDiffPipeline
    from animatediff.pipelines.pipeline_animation import AnimationPipeline
except Exception as e:
    logging.error("Error %s", str(e))


try:
    from latentflow.comfy_node import ComfyNode, ComfyResult
    from latentflow.comfy_ip_adapter_prompt_encode import ComfyIPAdapterPromptEncode
except Exception as e:
    logging.error("Error %s", str(e))


try:
    from latentflow.video_face_crop import VideoFaceCrop
except Exception as e:
    logging.error("Error %s", str(e))


try:
    from latentflow.video_rembg import VideoRembg
except Exception as e:
    logging.error("Error %s", str(e))

try:
    from latentflow.esrgan import RESRGANUpscale
except Exception as e:
    logging.error("Error %s", str(e))


try:
    from latentflow.sam import SegmentAnything, LoadSegmentAnything
except Exception as e:
    logging.error("Error %s", str(e))

try:
    from latentflow.dataset import Dataset
    from latentflow.train import TrainPredict, TrainCalcLoss, TrainStep
except Exception as e:
    logging.error("Error %s", str(e))


try:
    from latentflow.pipelines import TileRegionPipeline
except Exception as e:
    logging.error("Error %s", str(e))

