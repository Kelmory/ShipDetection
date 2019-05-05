import os
import cv2
import math
import numpy as np
from keras.models import load_model
from ShipDetection.net import *
from ShipDetection.data_io import CsvDirGenerator
import matplotlib.pyplot as plt
import skimage.measure as measure


def visualize_generator():
    gen = CsvDirGenerator(None).generate()
    for x, y in gen:
        plt.subplot(1, 2, 1).axis('off')
        x = x.reshape((768, 768))
        plt.imshow(x)
        plt.subplot(1, 2, 2).axis('off')
        y = y.reshape((768, 768))
        plt.imshow(y)
        plt.show()


def validate_loss(show=False):
    gen = CsvDirGenerator(None).generate()

    def focal_loss(y_true, y_pred, alpha=0.25, gamma=2):
        y_pred = np.clip(y_pred, 1e-9, 1 - 1e-9)
        pt_1 = np.where(np.equal(y_true, 1), y_pred, np.ones_like(y_pred))
        pt_0 = np.where(np.equal(y_true, 0), y_pred, np.zeros_like(y_pred))
        return -np.sum(alpha * np.power(1. - pt_1, gamma) * np.log(pt_1)) - \
               np.sum((1 - alpha) * np.power(pt_0, gamma) * np.log(1. - pt_0))

    for i, data in enumerate(gen):
        y = data[1]
        print('For img %d' % i)
        print('\t self: %f' % focal_loss(y, y))
        print('\t zero: %f' % focal_loss(y, np.zeros_like(y)))
        print('\t ones: %f' % focal_loss(y, np.ones_like(y)))
        print('\t rand: %f' % focal_loss(y, np.full_like(y, 0.5)))
        if show:
            x_ = data[0][0, :, :, 0] * 255
            y_ = y[0, :, :, 0] * 255
            edge = cv2.Canny(y_, 100, 300)
            edge = edge / 255.0
            plt.subplot(1, 1, 1).axis('off')
            plt.imshow(x_)
            plt.imshow(y_, alpha=0.4)
            plt.imshow(edge, alpha=0.8)
            plt.show()
        if i == 5:
            break


def visualize_predict(net_path, test_path, start=None, nums=None, post_process=False, mode='img'):
    try:
        net = load_model(net_path, compile=False)
        net.summary()
    except (ValueError, OSError):
        print('[Info]Failed to load model.')
        return

    if mode not in ['img', 'hist']:
        raise ValueError('`mode` not supported, should be `img` or `hist`.')

    path = test_path
    imgs = os.listdir(path)
    if all((start, nums)):
        end = start + nums
    else:
        nums = len(imgs)
        start, end = 0, nums
    imgs = imgs[start: end]
    imgs = [cv2.imread(path + '/' + img, cv2.IMREAD_GRAYSCALE) for img in imgs]
    imgs = np.asarray(imgs)
    imgs = imgs.reshape(imgs.shape)

    rows = int(math.sqrt(nums) / 2) * 2
    columns = math.ceil(nums * 2. / rows)
    for i in range(nums):
        x = imgs[i]
        plt.subplot(rows, columns, 2 * i + 1, title='Image').axis('off')
        x = x.reshape((768, 768))
        x = cv2.blur(x, (5, 5))
        plt.imshow(x)

        y = net.predict(x.reshape((1,) + x.shape + (1,)))
        y = y.reshape((768, 768))

        if mode is 'img':
            if post_process:
                y = cv2.medianBlur(y, 5)
                ratio = np.sum(y) / np.sum(np.ones_like(y))
                if ratio > 0.5:
                    y = 1.0 - y
            plt.subplot(rows, columns, 2 * i + 2, title='Predict').axis('off')
            plt.imshow(y)
        elif mode is 'hist':
            plt.subplot(rows, columns, 2 * i + 2, title='Histogram')
            plt.xlim((0, 1))
            plt.hist(y.ravel(), bins=20, log=True)
    plt.show()


def get_slices(img: np.ndarray) -> list:
    label = measure.label(img)
    label = measure.regionprops(label)

    return label


if __name__ == '__main__':
    # validate_loss()
    # visualize_generator()
    visualize_predict('E:/Data/ShipDetection/FCN/UNet_Any_epoch10.h5',
                      'E:/Data/ShipDetection/FCN/samples', mode='img', post_process=False)
