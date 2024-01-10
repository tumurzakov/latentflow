import logging
from .flow import Flow

class State(Flow):
    def __init__(self):
        super().__init__()
        self.state = {}
        self.key = None
        logging.debug("State init")

    def __call__(self, *args):
        key = None
        if isinstance(args[0], tuple):
            key = args[0][0]
        elif isinstance(args[0], str):
            key = args[0]
        else:
            raise Exception("Unknown key")

        self.key = key

        return self

    def set(self, value):
        self[self.key] = value

    def __setitem__(self, key, value):
        logging.debug("State set %s", key)
        self.state[key] = value

    def __getitem__(self, key):
        if key in self.state:
            return self.state[key]
        return None

    def __str__(self):
        return f'State({self.state.keys()})'

