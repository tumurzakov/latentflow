import logging

class Flow:

    def __invert__(self):
        return self

    def __or__(self, other):
        assert hasattr(other, 'apply'), f'{type(other)} should have apply mehtod'
        logging.debug('Flow pipe %s.apply(%s)', type(other), type(self))
        return other.apply(self)

    def __sub__(self, other):
        logging.debug('Flow pipe bypass %s.apply(%s)', type(other), type(self))
        return self

    def __lt__(self, other):
        logging.debug('Flow set bypass %s.set(%s)', type(other), type(self))
        return self

    def __gt__(self, other):
        assert hasattr(other, 'set'), f'{type(other)} should have set mehtod'
        logging.debug('Flow set %s.set(%s) -> %s', type(other), type(self), type(self))
        other.set(self)
        return self

    def __rshift__(self, other):
        return other

    def apply(self, other):
        return other

class If(Flow):
    def __init__(self, callback, if_true, if_false):
        self.callback = callback
        self.if_true = if_true
        self.if_false = if_false
        logging.debug("If init")

    def apply(self, other):
        cond = self.callback()
        logging.debug("If apply %s", cond)
        return self.if_true.apply(other) if cond else self.if_false

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
