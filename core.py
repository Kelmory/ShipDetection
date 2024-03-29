from ShipDetection.data_io import *
from ShipDetection.net import *


class Config:
    config = {}

    def sync_from_dict(self, config_dict):
        if not isinstance(config_dict, dict):
            raise TypeError('`config_dict` should be a dict.')
        for option in config_dict:
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
        func = getattr(self.net, mode)
        func(generator=self.generator.generate(), valid_generator=self.generator.generate(valid=True), **kwargs)


class CoreHolder:
    daemon = None
