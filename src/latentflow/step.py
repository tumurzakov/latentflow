import torch
import logging

from .flow import Flow
from .state import State

class Step(Flow):
    def __init__(self, scheduler, timestep, noise_predict, latent):
        self.scheduler = scheduler
        self.timestep = timestep
        self.noise_predict = noise_predict
        self.latent = latent

        logging.debug("Step init")

    def apply(self, other) -> State:

        # compute the previous noisy sample x_t -> x_t-1
        self.latent.latent[:] = self.scheduler.step(
                self.noise_predict.latent,
                self.timestep,
                self.latent.latent,
                ).prev_sample

        logging.debug("Step apply %s", self.latent)

        return self.latent

