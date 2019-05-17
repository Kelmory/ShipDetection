import os
import cv2
import math
import numpy as np
import pandas as pd


def normalize(array_obj: np.ndarray, padding=None, resize=None, reverse=False) -> np.ndarray:
    new = []
    for sample in array_obj:
        if isinstance(padding, tuple):
            res = np.zeros(padding)
            res[:sample.shape[0], :sample.shape[1], :] = sample
            sample = res
        if isinstance(resize, tuple):
            if len(resize) == 3:
                resize = resize[:2]
            if all(resize):
                sample = cv2.resize(sample, resize)
        if len(sample.shape) < 3:
            sample = np.reshape(sample, sample.shape + (1,))
        if reverse:
            sample = np.ones_like(sample) - sample
        new.append(sample)
    return np.asarray(new)


class PathHelper:
    def __init__(self, path):
        self.path = path

    def is_dir_valid(self):
        if not isinstance(self.path, str):
            return False
        else:
            return os.path.isdir(self.path)

    def is_file_valid(self):
        if not isinstance(self.path, str):
            return False
        else:
            return os.path.isfile(self.path)


class DataGenerator:
    _parent_path = 'E:/Data/ShipDetection/FCN'
    _load_mode = 'disk'
    _fetch_modes = ('disk', 'memory')
    _data_loaded = False
    _batch_size = 1
    _divided = False
    _gen_mode = ('train', 'valid', 'test')
    _use_mode = 'test'
    _steps = 10
    _dropna = False
    _divide_k = 8
    _x_shape = (768,768,1)
    _y_shape = (768,768,1)
    _augmentation = False
    _refer: pd.DataFrame = None

    def __init__(self, config=None):
        if config:
            self.config_sync(config)

        if self._load_mode not in self._fetch_modes:
            raise KeyError('Invalid `load_mode`, should be `disk` or `memory`.')

        if self._use_mode not in self._gen_mode:
            raise KeyError('Invalid `use_mode`, should be `train`, `valid`, or `test`')

        if not PathHelper(self._parent_path).is_dir_valid():
            raise KeyError('Invalid `parent_path`, check the path.')

    def load_data(self):
        pass

    def divide(self) -> (pd.DataFrame, pd.DataFrame):
        pass

    def generate(self):
        yield None, None

    def get_step(self):
        return self._steps

    def config_sync(self, config):
        for setting in config.config:
            if hasattr(self, setting):
                self.__setattr__(setting, config.config[setting])


class BinDirDataGenerator(DataGenerator):
    _parent_path = 'E:/Data/ShipDetection/CNN'
    _pos_path = 'ship'
    _neg_path = 'negative'

    def __init__(self, config):
        super().__init__(config)

        # path of negative and positive samples
        positive = self._parent_path + '/' + self._pos_path
        negative = self._parent_path + '/' + self._neg_path

        if PathHelper(negative).is_dir_valid():
            self._neg_path = negative
        else:
            raise ValueError('Negative sample path should be `str` and valid but %s found.' % negative)

        if PathHelper(positive).is_dir_valid():
            self._pos_path = positive
        else:
            raise ValueError('Positive sample path should be `str` and valid but %s found.' % positive)

        self.load_data()

    def load_data(self):
        pos_files = os.listdir(self._pos_path)
        neg_files = os.listdir(self._neg_path)

        pos = {self._pos_path + '/' + name: 1 for name in pos_files}
        neg = {self._neg_path + '/' + name: 0 for name in neg_files}

        pos = pd.DataFrame.from_dict(pos, orient='index')
        neg = pd.DataFrame.from_dict(neg, orient='index')

        refer = pos.append(neg).reset_index()
        refer.columns = ['ImageId', 'IsShip']
        self._refer = refer.sample(frac=1).reset_index(drop=True)[:self._steps]

        if self._load_mode is 'disk':
            self._refer['x'] = self._refer['ImageId']
            self._refer['y'] = self._refer['IsShip']
        elif self._load_mode is 'memory':
            self._refer['ImageId'] = self._refer['ImageId'].apply(lambda img: cv2.imread(img, cv2.IMREAD_GRAYSCALE))
            self._refer['ImageId'] = self._refer['ImageId'].apply(lambda img: np.reshape(img, img.shape + (1,)) / 255.0)
            self._refer['x'] = self._refer['ImageId']
            self._refer['y'] = self._refer['IsShip']
        else:
            raise KeyError('read mode should be `disk` or `memory`, but %s found.' % self._load_mode)

        self._steps = math.ceil(len(self._refer['x']) / self._batch_size)
        self._data_loaded = True

    def generate(self, valid=False):
        if not self._data_loaded:
            raise Exception('Data unloaded, use `load_data` method before using `generate`.')

        x_, y_ = self.divide(valid)

        if not self._divided:
            raise Exception('Data not divided, use `divide` method.')

        i = 0
        while True:
            if self._load_mode is 'disk':
                x = x_[int(i * self._batch_size): int((i + 1) * self._batch_size)].apply(
                    lambda img: cv2.imread(img, cv2.IMREAD_GRAYSCALE))
                y = y_[int(i * self._batch_size): int((i + 1) * self._batch_size)]
            elif self._load_mode is 'memory':
                x = x_[int(i * self._batch_size): int((i + 1) * self._batch_size)]
                y = y_[int(i * self._batch_size): int((i + 1) * self._batch_size)]
            else:
                raise KeyError('Wrong generate mode, `disk` or `memory` expected, but %s found' % self._load_mode)
            x = normalize(x, resize=self._x_shape)
            y = np.asarray(y, dtype=np.float32)
            x = x / 255.0
            i = (i + 1) % math.ceil(len(x_) / self._batch_size)
            yield x, y

    def divide(self, valid=False):
        self._refer = self._refer.sample(frac=1).reset_index(drop=True)
        if self._use_mode not in self._gen_mode:
            raise KeyError('`mode` should be in  %s, but %s found.' % (self._gen_mode, self._use_mode))
        elif self._use_mode is 'test':
            self._divided = True
            return self._refer['x'], self._refer['y']
        elif self._use_mode is 'train' and not valid:
            x_ = self._refer['x'].drop(index=list(range(0, len(self._refer['x']), self._divide_k)))
            y_ = self._refer['y'].drop(index=list(range(0, len(self._refer['y']), self._divide_k)))
        elif valid:
            x_ = self._refer['x'][::self._divide_k]
            y_ = self._refer['y'][::self._divide_k]
        else:
            raise KeyError('`mode` should be in  %s, but %s found.' % (self._gen_mode, self._use_mode))
        self._divided = True
        return x_, y_


class CsvDirGenerator(DataGenerator):
    _csv_path = "train2.csv"
    _dir_path = "train"

    _bf_ratio = 2

    def __init__(self, config):
        super().__init__(config)

        csv_path = self._parent_path + '/' + self._csv_path
        dir_path = self._parent_path + '/' + self._dir_path

        if PathHelper(csv_path).is_file_valid():
            self._csv_path = csv_path
        else:
            raise ValueError('CSV path should be `str` and valid but %s found.' % csv_path)

        if PathHelper(dir_path).is_dir_valid():
            self._dir_path = dir_path
        else:
            raise ValueError('Dir path should be `str` and valid but %s found.' % dir_path)

        self.load_data()

    def load_data(self):
        refer = pd.read_csv(self._csv_path)
        if self._dropna:
            refer = refer.dropna()
        else:
            na_set = refer.loc[refer['EncodedPixels'].isna()].sample(frac=1).reset_index(drop=True)[:int(self._steps / self._bf_ratio)]
            pos_set = refer.dropna().sample(frac=1).reset_index(drop=True)[:self._steps - int(self._steps / self._bf_ratio)]
            refer = na_set.append(pos_set)
        self._refer = refer.sample(frac=1).reset_index(drop=True)[:self._steps]

        imread_param = cv2.IMREAD_GRAYSCALE if self._x_shape[-1] is 1 else cv2.IMREAD_COLOR

        if self._load_mode is 'memory':
            self._refer['x'] = self._refer['ImageId'].apply(
                lambda img_name: cv2.imread(self._dir_path + '/' + img_name), imread_param)
            self._refer['y'] = self._refer['EncodedPixels'].apply(lambda pixels: self.__mask_decode(pixels))
        elif self._load_mode is 'disk':
            self._refer['x'] = self._refer['ImageId'].apply(lambda img_name: self._dir_path + '/' + img_name)
            self._refer['y'] = self._refer['EncodedPixels']

        self._steps = math.ceil(len(self._refer['x']) / self._batch_size)
        self._data_loaded = True

    def generate(self, valid=False):
        if not self._data_loaded:
            raise Exception('Data unloaded, use `load_data` method before using `generate`.')

        x_, y_ = self.divide(valid)

        imread_param = cv2.IMREAD_GRAYSCALE if self._x_shape[-1] is 1 else cv2.IMREAD_COLOR

        if not self._divided:
            raise Exception('Data not divided, use `divide` method.')

        i = 0
        while True:
            if self._load_mode is 'disk':
                x = x_[int(i * self._batch_size): int((i + 1) * self._batch_size)].apply(
                    lambda img: cv2.imread(img, imread_param))
                y = y_[int(i * self._batch_size): int((i + 1) * self._batch_size)].apply(
                    lambda label: self.__mask_decode(label))
            elif self._load_mode is 'memory':
                x = x_[int(i * self._batch_size): int((i + 1) * self._batch_size)]
                y = y_[int(i * self._batch_size): int((i + 1) * self._batch_size)]
            else:
                raise KeyError('Wrong generate mode, `disk` or `memory` expected, but %s found' % self._load_mode)
            x = normalize(x, resize=self._x_shape)
            y = normalize(y, resize=self._y_shape)
            if self._augmentation:
                x = np.concatenate((x, x * np.random.random() * 1.5), axis=0)
                y = np.concatenate((y, y.copy()), axis=0)
            i = (i + 1) % math.ceil(len(x_) / self._batch_size)
            yield x, y

    def divide(self, valid=False):
        self._refer = self._refer.sample(frac=1).reset_index(drop=True)
        if self._use_mode not in self._gen_mode:
            raise KeyError('`mode` should be in  %s, but %s found.' % (self._gen_mode, self._use_mode))
        elif self._use_mode is 'test':
            self._divided = True
            return self._refer['x'], self._refer['y']
        elif self._use_mode is 'train' and not valid:
            x_ = self._refer['x'].drop(index=list(range(0, len(self._refer['x']), self._divide_k)))
            y_ = self._refer['y'].drop(index=list(range(0, len(self._refer['y']), self._divide_k)))
        elif valid:
            x_ = self._refer['x'][::self._divide_k]
            y_ = self._refer['y'][::self._divide_k]
        else:
            raise KeyError('`mode` should be in  %s, but %s found.' % (self._gen_mode, self._use_mode))
        self._divided = True
        return x_, y_

    def __mask_decode(self, element):
        mask = np.zeros((768, 768), dtype=np.uint8).flatten()
        if isinstance(element, str):
            element = element.split(' ')
            start, steps = np.asarray(element[0::2], dtype=np.int32), np.asarray(element[1::2], dtype=np.int32)
            end = start + steps - 1
            for lo, hi in zip(start, end):
                mask[lo: hi] = 1
        mask = np.transpose(mask.reshape((768, 768)), axes=[1, 0])
        return mask

