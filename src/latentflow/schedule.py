import torch
import logging
from typing import Callable, List, Optional, Tuple, Union, Generator

from diffusers import (
        EulerDiscreteScheduler, DDIMScheduler, DPMSolverMultistepScheduler,
        EulerAncestralDiscreteScheduler, HeunDiscreteScheduler
        )

SchedulerInput = Union[
        EulerDiscreteScheduler,
        DDIMScheduler,
        DPMSolverMultistepScheduler,
        EulerAncestralDiscreteScheduler,
        HeunDiscreteScheduler
]

from .flow import Flow
from .state import State
from .timesteps import Timesteps

class Schedule(Flow):
    def __init__(self,
            scheduler:Optional[SchedulerInput]=None,
            strength:Optional[float] = None,
            num_inference_steps: Optional[int] = None,
            onload_device: str='cuda',
            offload_device: str='cpu',
            ):

        self.scheduler = scheduler
        self.strength = strength
        self.num_inference_steps = num_inference_steps
        self.onload_device = onload_device
        self.offload_device = offload_device

    def calc_strength_timesteps(self, num_inference_steps, strength, device):
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]
        return timesteps, num_inference_steps - t_start

    def apply(self, other) -> Timesteps:

        self.scheduler.set_timesteps(self.num_inference_steps, device=self.onload_device)
        self.timesteps = self.scheduler.timesteps
        self.num_inference_steps = len(self.timesteps)

        if self.strength is not None:
            self.timesteps, self.num_inference_steps = self.calc_strength_timesteps(
                    self.num_inference_steps,
                    self.strength,
                    self.onload_device)

        result = Timesteps(self.timesteps, self.num_inference_steps)

        result.offload()

        return result
