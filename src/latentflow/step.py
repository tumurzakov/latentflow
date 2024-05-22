import torch
import logging

from .flow import Flow
from .state import State
from .debug import tensor_hash
from .latent import NoisePredict


class Step(Flow):
    def __init__(self,
            scheduler,
            timestep,
            noise_predict=None,
            latent=None,
            ):
        self.scheduler = scheduler
        self.timestep = timestep
        self.noise_predict = noise_predict
        self.latent = latent

        logging.debug("Step init %s %s %s",
                timestep, noise_predict, latent)

    def apply(self, noise_predict: NoisePredict):

        noise_predict.onload()
        self.latent.onload()

        latent_model_input = self.latent.latent
        latent_model_input = self.scheduler.scale_model_input(latent_model_input, self.timestep)

        if noise_predict is None and isinstance(self.noise_predict, NoisePredict):
            noise_predict = self.noise_predict

        # compute the previous noisy sample x_t -> x_t-1
        self.latent.latent[:] = self.scheduler.step(
                noise_predict.latent,
                self.timestep,
                latent_model_input,
                ).prev_sample

        logging.debug("Step apply %s %s %s",
                self.timestep, noise_predict, self.latent)

        self.latent.offload()
        noise_predict.offload()

        return self.latent

