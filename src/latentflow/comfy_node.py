import torch
import logging

from .flow import Flow

class ComfyResult(Flow):
    def __init__(self, result, return_types):
        self.result = {}
        self.return_types = return_types

        for i, t in enumerate(return_types):
            self.result[t] = result[i]

    def __getitem__(self, key):
        return self.result[key]

    def __str__(self):
        rep = []
        for k in self.result:
            rep.append(f"{k}: {type(self.result[k])}")

        return "ComfyResult(%s)" % ",".join(rep)

class ComfyNode(Flow):
    def __init__(self, comfy_node_cls, **kwargs):
        self.comfy_node_cls = comfy_node_cls
        self.kwargs = kwargs

        self.instance = comfy_node_cls()

    def apply(self, other=None):

        input_types = self.comfy_node_cls.INPUT_TYPES()
        for k in input_types:
            for f in input_types[k]:

                input_values = {}
                if f in self.kwargs:
                    if not isinstance(self.kwargs[f], ComfyResult):
                        continue
                    input_values = self.kwargs[f].result

                o = {}
                if len(input_types[k][f]) == 1:
                    t, = input_types[k][f]
                else:
                    t, o = input_types[k][f]

                if isinstance(t, list):
                    self.kwargs[f] = t[0]
                elif t in input_values:
                    self.kwargs[f] = input_values[t]
                elif t in other.result:
                    self.kwargs[f] = other[t]
                elif 'default' in o:
                    self.kwargs[f] = o['default']

        result = getattr(self.instance, self.comfy_node_cls.FUNCTION)(**self.kwargs)

        return ComfyResult(result, self.comfy_node_cls.RETURN_TYPES)
