import os
import cv2
import numpy as np
from keras.models import load_model
from ShipDetection.net import *
from ShipDetection.data_io import CsvDirGenerator
import matplotlib.pyplot as plt


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


def visualize_predict(start=0, n=20):
    net = DenoiseNet(None).get_model()
    net.summary()
    try:
        net.load_weights('E:/Data/ShipDetection/FCN/model.h5')
    except ValueError:
        print('[Info]Failed to load model.')
        exit(0)

    path = 'E:/Data/ShipDetection/FCN/samples'
    imgs = os.listdir(path)[start: start + n]
    imgs = [cv2.imread(path + '/' + img, cv2.IMREAD_GRAYSCALE) for img in imgs]
    imgs = np.asarray(imgs)
    imgs = imgs.reshape(imgs.shape + (1,))

    for i in range(n):
        x = imgs[i]
        y = net.predict(x.reshape((1,) + x.shape))
        plt.subplot(5, n * 0.4, 2 * i + 1).axis('off')
        x = x.reshape((768, 768))
        plt.imshow(x)
        plt.subplot(5, n * 0.4, 2 * i + 2).axis('off')
        y = y.reshape((768, 768))
        y = cv2.equalizeHist(np.uint8(y * 255))
        plt.imshow(y)
        # plt.hist(y.ravel(), 20)
    plt.show()


def migrate_model(model_path, model_name):
    model_path = 'E:/Data/ShipDetection/FCN' if model_path is None else model_path
    model_name = model_path + '/' + model_name


if __name__ == '__main__':
    # validate_loss()
    # visualize_generator()
    visualize_predict()
