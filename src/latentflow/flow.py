import logging

class Flow:

    def __invert__(self):
        return self

    def __or__(self, other):
        assert hasattr(other, 'apply'), f'{type(other)} should have apply mehtod'
        logging.debug('Flow pipe %s.apply(%s)', type(other), type(self))
        return other.apply(self)

    def __sub__(self, other):
        return self

    def __lt__(self, other):
        return self

    def __gt__(self, other):
        assert hasattr(other, 'set'), f'{type(other)} should have set method'
        logging.debug('Flow set %s.set(%s) -> %s', type(other), type(self), type(self))
        other.set(self)
        return self

    def __rshift__(self, other):
        return other

    def apply(self, other):
        return other

class If(Flow):
    def __init__(self, condition, callback):
        self.condition = condition
        self.callback = callback

    def apply(self, other):
        if self.condition:
            return self.callback(other)
        else:
            return other

class Set(Flow):
    def __init__(self, var, key=None):
        self.var = var
        self.key = key

    def apply(self, other):
        if self.key is None:
            self.var.set(other)
        else:
            self.var[self.key] = other
        return other
