import logging
from .flow import Flow
import gc
import torch

class State(Flow):
    def __init__(self, state: dict = {}):
        super().__init__()
        self.state = state
        self.key = None
        logging.debug("State init")

    def __call__(self, *args):
        assert len(args) > 0, 'State, not enought arguments'
        self.key = args[0]
        return self

    def update(self, state):
        self.state.update(state)
        return self

    def clone(self):
        clone_state = State({})
        clone_state.update(self.state)
        return clone_state

    def clear(self, key):
        del self.state[key]
        gc.collect()
        torch.cuda.empty_cache()
        return self

    def set(self, value):

        if isinstance(self.key, tuple):
            # for simplicity
            assert len(self.key) == 2, "Should contain only two args"
            key = self.key[0]
            idx = self.key[1]

            logging.debug("State set %s %s", key, idx)

            if isinstance(idx, str):
                if key not in self.state:
                    self[key] = {}

                self[key][str(idx)] = (idx, value)

                logging.debug("State hash %s %s", key, idx)

            elif isinstance(idx, int):
                if key not in self.state:
                    self[key] = []

                if idx >= len(self[key]):
                    for i in range(len(self[key]), idx+1):
                        self[key].append(None)

                self[key][idx] = value

                logging.debug("State list %s %s", key, idx)

        elif isinstance(self.key, str):
            logging.debug("State set %s %s", self.key, type(value))
            self[self.key] = value

        else:
            raise TypeError("Not valid type")

    def __setitem__(self, key, value):
        self.state[key] = value

    def __getitem__(self, key):
        if key in self.state:
            return self.state[key]
        return None

    def __str__(self):
        return f'State({self.state.keys()})'


class Context(State): ...

class Free(Flow):
    def __init__(self, state, key):
        self.state = state
        self.key = key

    def apply(self, other):
        self.state[self.key] = None
        gc.collect()
        return other
