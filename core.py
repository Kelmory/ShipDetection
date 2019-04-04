from ShipDetection.data_io import *
from ShipDetection.net import *

import multiprocessing


class Config:
    config = {}

    def __init__(self):
        self.config['_epoch'] = 100
        self.config['_batch_size'] = 8
        self.config['_lr'] = 1e-4
        self.config['_model_path'] = ''
        self.config['_steps'] = 1
        self.config['_parent_path'] = ''
        self.config['_load_mode'] = None
        self.config['_use_mode'] = None
        self.config['_divide_k'] = 8
        self.config['_x_shape'] = None
        self.config['_y_shape'] = None
        self.config['_neg_path'] = None
        self.config['_pos_path'] = None
        self.config['_csv_path'] = None
        self.config['_dir_path'] = None

    def sync_from_dict(self, config_dict):
        if not isinstance(config_dict, dict):
            raise TypeError('`config_dict` should be a dict.')
        for option in config_dict:
            if option in self.config.keys():
                self.config[option] = config_dict[option]


class Pipeline:
    def __init__(self, net, generator, configs):
        self.config = Config()
        self.config.sync_from_dict(configs)

        if not isinstance(net, str):
            raise TypeError('Net should be a instance of `Net`.')
        if not isinstance(generator, str):
            raise TypeError('Generator should be a instance of `DataGenerator`.')

        if generator not in [i.__name__ for i in DataGenerator.__subclasses__()]:
            raise ValueError('`generator` name error, check the implementation.')
        else:
            generator_module = __import__('data_io')
            self.generator = getattr(generator_module, generator)(self.config)
            self.config.config['_steps'] = self.generator.get_step()

        if net not in [i.__name__ for i in Net.__subclasses__()]:
            raise ValueError('`net` name error, check the implementation.')
        else:
            net_module = __import__('net')
            self.net = getattr(net_module, net)(self.config)

    def work(self, mode, **kwargs):
        if mode not in ('train', 'predict'):
            raise KeyError('`mode` should be `train` or `predict`.')
        else:
            func = getattr(self.net, mode)
            func(generator=self.generator.generate(), **kwargs)


class CoreHolder:
    daemon = None

    def __init__(self):
        self.daemon = multiprocessing.process.active_children()
