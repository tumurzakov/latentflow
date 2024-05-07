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
    def __init__(self, condition, callback, desc="", raise_errors=True):
        self.condition = condition
        self.callback = callback
        self.desc = desc
        self.raise_errors = raise_errors

    def apply(self, other):
        logging.debug("If %s %s", self.condition, self.desc)
        if self.condition:
            if self.raise_errors:
                return self.callback(other)

            try:
                return self.callback(other)
            except Exception as e:
                logging.error("If error %s", e)
                return other
        else:
            return other

class Switch(Flow):
    def __init__(self, callback, if_true, if_false):
        self.callback = callback
        self.if_true = if_true
        self.if_false = if_false
        logging.debug("Switch init")

    def apply(self, other):
        cond = self.callback(other)
        logging.debug("Switch apply %s", cond)
        return self.if_true.apply(other) if cond else self.if_false.apply(other)


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
