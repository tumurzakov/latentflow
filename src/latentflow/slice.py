import logging
def slice_scale(s, q):
    result = None
    if isinstance(s, slice):
        result = slice(s.start*q, s.stop*q)
    elif isinstance(s, list):
        l = []
        for x in s:
            for i in range(q):
                l.append(x*q + i)
        result = l
    else:
        raise TypeError(f"Uknown type {type(s)}")

    return result
