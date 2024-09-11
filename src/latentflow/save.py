import torch
import logging
import os
import datetime
import shutil


from .flow import Flow

class Save(Flow):
    def __init__(self, path, **kwargs):
        self.path = path
        self.kwargs = kwargs

    def apply(self, other):
        assert hasattr(other, 'save'), f'{type(other)} must have save method'

        if '%datetime%' in self.path:
            self.path = self.path.replace('%datetime%',
                    datetime.datetime.now().strftime("%FT%T"))

        logging.info('Save %s', self.path)

        dir_path = os.path.dirname(self.path)
        os.makedirs(dir_path, exist_ok=True)

        if 'additional_files' in self.kwargs:
            for file in self.kwargs['additional_files']:
                shutil.copyfile(file, dir_path)

        other.save(self.path, **self.kwargs)
        return other

class Load(Flow):
    def __init__(self, path, **kwargs):
        self.path = path
        self.kwargs = kwargs

    def apply(self, other):
        assert hasattr(other, 'load'), f'{type(other)} must have load method'
        return other.load(self.path, **self.kwargs)

