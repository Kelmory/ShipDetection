import os
import cv2
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def normalize(array_obj: np.ndarray, padding=None) -> np.ndarray:
    new = []
    for sample in array_obj:
        if isinstance(padding, tuple):
            res = np.zeros(padding)
            res[:sample.shape[0],:sample.shape[1], :] = sample
            sample = res
        new.append(sample)
    return np.asarray(new)


def visualize(x, y):
    plt.subplot(1, 2, 1).axis('off')
    x = x.reshape((768, 768))
    plt.imshow(x)
    plt.subplot(1, 2, 2).axis('off')
    y = y.reshape((768, 768))
    plt.imshow(y)
    plt.show()


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
    _steps = 1000
    _divide_k = 8
    _x_shape = (768,768,1)
    _y_shape = (768,768,1)

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

    def divide(self):
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
    _pos_path = ''
    _neg_path = ''
    _pos = []
    _neg = []

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

        if self._load_mode is 'disk':
            self._pos = [(self._pos_path + '/' + img) for img in pos_files]
            self._neg = [(self._neg_path + '/' + img) for img in neg_files]
        elif self._load_mode is 'memory':
            self._pos = [cv2.imread(self._pos_path + '/' + img, cv2.IMREAD_GRAYSCALE) for img in pos_files]
            self._neg = [cv2.imread(self._neg_path + '/' + img, cv2.IMREAD_GRAYSCALE) for img in neg_files]
            self._pos = [np.reshape(img, img.shape + (1,)) / 255.0 for img in self._pos]
            self._neg = [np.reshape(img, img.shape + (1,)) / 255.0 for img in self._neg]
        else:
            raise KeyError('read mode should be `disk` or `memory`, but %s found.' % self._load_mode)

        self._steps = (len(self._pos) + len(self._neg)) - math.ceil((len(self._pos) + len(self._neg)) / self._divide_k)

        self._data_loaded = True

    def generate(self):
        if not self._data_loaded:
            raise Exception('Data unloaded, use `load_data` method before using `generate`.')

        pos, neg = self.divide()
        total = len(pos) + len(neg)

        if not self._divided:
            raise Exception('Data not divided, use `divide` method.')

        pos_per_batch = round(len(pos) / total * self._batch_size)
        neg_per_batch = round(len(neg) / total * self._batch_size)

        i = 0
        while True:
            if self._load_mode is 'disk':
                x_pos = [np.array(cv2.imread(img, cv2.IMREAD_GRAYSCALE))
                         for img in pos[i * pos_per_batch: (i + 1) * pos_per_batch]]
                x_neg = [np.array(cv2.imread(img, cv2.IMREAD_GRAYSCALE))
                         for img in neg[i * neg_per_batch: (i + 1) * neg_per_batch]]
                x_pos = [np.reshape(img, img.shape + (1,)) for img in x_pos]
                x_neg = [np.reshape(img, img.shape + (1,)) for img in x_neg]
            elif self._load_mode is 'memory':
                x_pos = pos[i * pos_per_batch: (i + 1) * pos_per_batch]
                x_neg = neg[i * neg_per_batch: (i + 1) * neg_per_batch]
            else:
                raise KeyError('Wrong generate mode, `disk` or `memory` expected, but %s found' % self._load_mode)
            x = np.array((x_pos + x_neg))
            x = normalize(x, padding=(200, 200, 1))
            x = x / 255.0
            y = np.array([1.0] * pos_per_batch + [0.0] * neg_per_batch)
            i = (i + 1) % self._steps
            yield x, y

    def divide(self):
        if self._use_mode not in self._gen_mode:
            raise KeyError('`mode` should be in  %s, but %s found.' % (self._gen_mode, self._use_mode))
        elif self._use_mode is 'test':
            self._divided = True
            return self._pos, self._neg
        elif self._use_mode is 'train':
            pos = [i if self._pos.index(i) % self._divide_k != 0 else None for i in self._pos]
            neg = [i if self._neg.index(i) % self._divide_k != 0 else None for i in self._neg]
        else:
            pos = [i if self._pos.index(i) % self._divide_k == 0 else None for i in self._pos]
            neg = [i if self._neg.index(i) % self._divide_k == 0 else None for i in self._neg]
        while None in pos:
            pos.remove(None)
        while None in neg:
            neg.remove(None)
        self._divided = True
        return pos, neg


class CsvDirGenerator(DataGenerator):
    _csv_path = "train2.csv"
    _dir_path = "train"
    _refer = None
    _x_ = None
    _y_ = None

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
        self._refer = refer.sample(frac=1).reset_index(drop=True)[:2000]

        if self._load_mode is 'memory':
            self._x_ = self._refer['ImageId'].apply(lambda img_name: cv2.imread(self._dir_path + '/' + img_name))
            self._y_ = self._refer['EncodedPixels'].apply(lambda pixels: self.__mask_decode(pixels))
        elif self._load_mode is 'disk':
            self._x_ = self._refer['ImageId'].apply(lambda img_name: self._dir_path + '/' + img_name)
            self._y_ = self._refer['EncodedPixels']

        self._steps = len(self._x_) - math.ceil(len(self._x_) / self._divide_k)
        self._data_loaded = True

    def generate(self):
        if not self._data_loaded:
            raise Exception('Data unloaded, use `load_data` method before using `generate`.')

        x_, y_ = self.divide()

        if not self._divided:
            raise Exception('Data not divided, use `divide` method.')

        i = 0
        while True:
            if self._load_mode is 'disk':
                x = x_[i * self._batch_size: (i + 1) * self._batch_size].apply(
                    lambda img: cv2.imread(img, cv2.IMREAD_GRAYSCALE))
                y = y_[i * self._batch_size: (i + 1) * self._batch_size].apply(
                    lambda label: self.__mask_decode(label))
            elif self._load_mode is 'memory':
                x = x_[i * self._batch_size: (i + 1) * self._batch_size]
                y = y_[i * self._batch_size: (i + 1) * self._batch_size]
            else:
                raise KeyError('Wrong generate mode, `disk` or `memory` expected, but %s found' % self._load_mode)
            x = normalize(x)
            x = x.reshape(x.shape + (1,))
            y = normalize(y)
            x = x / 255.0
            i = (i + 1) % self._steps
            yield x, y

    def divide(self):
        if self._use_mode not in self._gen_mode:
            raise KeyError('`mode` should be in  %s, but %s found.' % (self._gen_mode, self._use_mode))
        elif self._use_mode is 'test':
            self._divided = True
            return self._x_, self._y_
        elif self._use_mode is 'train':
            x_ = self._x_.drop(index=list(range(0, len(self._x_), self._divide_k)))
            y_ = self._y_.drop(index=list(range(0, len(self._y_), self._divide_k)))
        else:
            x_ = self._x_[::self._divide_k]
            y_ = self._y_[::self._divide_k]
        self._divided = True
        return x_, y_

    def __mask_decode(self, element):
        mask = np.zeros(self._y_shape, dtype=np.uint8).flatten()
        if isinstance(element, str):
            element = element.split(' ')
            start, steps = np.asarray(element[0::2], dtype=np.int32), np.asarray(element[1::2], dtype=np.int32)
            end = start + steps - 1
            for lo, hi in zip(start, end):
                mask[lo: hi] = 1
        mask = mask.reshape(self._y_shape).T.reshape(self._y_shape)
        return mask


if __name__ == '__main__':
    gen = CsvDirGenerator(None)
    for x, y in gen.generate():
        visualize(x, y)
