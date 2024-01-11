import logging

class Flow:

    def __invert__(self):
        return self

    def __or__(self, other):
        assert hasattr(other, 'apply'), f'{type(other)} should have apply mehtod'
        return other.apply(self)

    def __sub__(self, other):
        logging.debug('Flow bypass %s', type(other))
        return self

    def __lt__(self, other):
        logging.debug('Flow bypass %s', type(other))
        return self

    def __gt__(self, other):
        assert hasattr(other, 'set'), f'{type(other)} should have set mehtod'
        other.set(self)

        return self

    def __rshift__(self, other):
        return other

    def apply(self, other):
        return other
