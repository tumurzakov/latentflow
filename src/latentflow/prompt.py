import logging

from .flow import Flow

class Prompt(Flow):
    def __init__(self, prompt: str = "", negative_prompt: str = ""):
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        logging.debug(f"init {self}")

    def __str__(self):
        return f'Prompt(+[{self.prompt}], -[{self.negative_prompt}])'
