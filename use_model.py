import os
import cv2
import numpy as np
from ShipDetection.net import UNet
import matplotlib.pyplot as plt


if __name__ == '__main__':
    net = UNet(None).get_model()
    net.summary()
    net.load_weights('E:/Data/ShipDetection/FCN/model.h5')

    path = 'E:/Data/ShipDetection/FCN/test'
    imgs = os.listdir(path)[0:20]
    imgs = [cv2.imread(path + '/' + img, cv2.IMREAD_GRAYSCALE) for img in imgs]
    imgs = np.asarray(imgs)
    imgs = imgs.reshape(imgs.shape + (1,))

    for i in range(20):
        x = imgs[i]
        y = net.predict(x.reshape((1,) + x.shape))
        plt.subplot(5, 8, 2 * i + 1).axis('off')
        plt.subplot(5, 8, 2 * i + 1).set_title('Image')
        x = x.reshape((768, 768))
        plt.imshow(x)
        plt.subplot(5, 8, 2 * i + 2).axis('off')
        plt.subplot(5, 8, 2 * i + 2).set_title('Predict')
        y = y.reshape((768, 768))
        # y = np.ones_like(y) - y
        plt.imshow(y)
    plt.show()
