import logging
from .flow import Flow

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

    def set(self, value):
        if isinstance(self.key, tuple):
            # for simplicity
            assert len(self.key) == 2, "Should contain only two args"
            key = self.key[0]
            idx = self.key[1]

            if key not in self:
                self[key] = {}

            self[key][str(idx)] = (idx, value)

        elif isinstance(self.key, str):
            self[self.key] = value

        else:
            raise TypeError("Not valid type")

    def __setitem__(self, key, value):
        logging.debug("State set %s", key)
        self.state[key] = value

    def __getitem__(self, key):
        if key in self.state:
            return self.state[key]
        return None

    def __str__(self):
        return f'State({self.state.keys()})'

