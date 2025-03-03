import os.path as osp
import sys
from importlib import import_module
from pathlib import PurePath

import yaml


class Config(object):
    """A facility for config and config files.

    It supports common file formats as configs: python/json/yaml. The interface
    is the same as a dict object and also allows access config values as
    attributes.
    """

    @staticmethod
    def fromfile(filename):
        if isinstance(filename, PurePath):
            filename = filename.as_posix()
        filename = osp.abspath(osp.expanduser(filename))
        if not osp.isfile(filename):
            raise KeyError("file {} does not exist".format(filename))
        if filename.endswith(".py"):
            module_name = osp.basename(filename)[:-3]
            if "." in module_name:
                raise ValueError("Dots are not allowed in config file path.")
            config_dir = osp.dirname(filename)

            old_module = None
            if module_name in sys.modules:
                old_module = sys.modules.pop(module_name)

            sys.path.insert(0, config_dir)
            mod = import_module(module_name)
            sys.path.pop(0)
            cfg_dict = {
                name: value
                for name, value in mod.__dict__.items()
                if not name.startswith("__")
            }
            # IMPORTANT: pop to avoid `import_module` from cache, to avoid the
            # cfg sharing by multiple processes or functions, which may cause
            # interference and get unexpected result.
            sys.modules.pop(module_name)

            if old_module is not None:
                sys.modules[module_name] = old_module

        elif filename.endswith((".yml", ".yaml")):
            with open(filename, "r") as fid:
                cfg_dict = yaml.load(fid, Loader=yaml.Loader)
        else:
            raise IOError(
                "Only py/yml/yaml type are supported now, "
                f"but found {filename}!"
            )
        return Config(cfg_dict, filename=filename)

    def __init__(self, cfg_dict=None, filename=None, encoding="utf-8"):
        if cfg_dict is None:
            cfg_dict = {}
        elif not isinstance(cfg_dict, dict):
            raise TypeError(
                "cfg_dict must be a dict, but got {}".format(type(cfg_dict))
            )

        super(Config, self).__setattr__("_cfg_dict", cfg_dict)
        super(Config, self).__setattr__("_filename", filename)
        if filename:
            with open(filename, "r", encoding=encoding) as f:
                super(Config, self).__setattr__("_text", f.read())
        else:
            super(Config, self).__setattr__("_text", "")

    @property
    def filename(self):
        return self._filename

    @property
    def text(self):
        return self._text

    def __repr__(self):
        return "Config (path: {}): {}".format(
            self.filename, self._cfg_dict.__repr__()
        )

    def __len__(self):
        return len(self._cfg_dict)

    def __getattr__(self, name):
        try:
            return getattr(self._cfg_dict, name)
        except AttributeError as e:
            if isinstance(self._cfg_dict, dict):
                try:
                    return self.__getitem__(name)
                except KeyError:
                    raise AttributeError(name)
            raise e

    def __getitem__(self, name):
        return self._cfg_dict.__getitem__(name)

    def __setattr__(self, name, value):
        self._cfg_dict.__setitem__(name, value)

    def __setitem__(self, name, value):
        self._cfg_dict.__setitem__(name, value)

    def __iter__(self):
        return iter(self._cfg_dict)
