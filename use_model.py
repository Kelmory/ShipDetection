import os
import cv2
import numpy as np
from keras.models import load_model, Model
from keras.layers import Input, BatchNormalization, \
    SeparableConv2D, MaxPooling2D, Conv2D, Concatenate, Deconv2D, UpSampling2D, ZeroPadding2D
import matplotlib.pyplot as plt


def _build():
    fcn_in = Input(shape=(768, 768, 1))

    # convolution
    bn1 = fcn_in  # BatchNormalization()(fcn_in)
    conv11 = SeparableConv2D(64, 3)(bn1)
    conv12 = SeparableConv2D(64, 3)(conv11)
    conv13 = MaxPooling2D()(conv12)

    bn2 = conv13  # BatchNormalization()(conv13)
    conv21 = SeparableConv2D(128, 3)(bn2)
    conv22 = SeparableConv2D(128, 3)(conv21)
    conv23 = MaxPooling2D()(conv22)

    bn3 = conv23  # BatchNormalization()(conv23)
    conv31 = SeparableConv2D(256, 3)(bn3)
    conv32 = SeparableConv2D(256, 3)(conv31)
    conv33 = MaxPooling2D()(conv32)

    bn4 = conv33  # BatchNormalization()(conv33)
    conv41 = SeparableConv2D(512, 3)(bn4)
    conv42 = SeparableConv2D(512, 3)(conv41)
    conv43 = MaxPooling2D()(conv42)

    bn5 = conv43  # BatchNormalization()(conv43)
    conv51 = SeparableConv2D(1024, 3)(bn5)
    conv52 = SeparableConv2D(1024, 3)(conv51)

    # deconvolution
    deconv51 = Deconv2D(128, 3)(conv52)
    deconv52 = Deconv2D(128, 3)(deconv51)
    deconv53 = UpSampling2D()(deconv52)

    concat4 = Concatenate()([deconv53, conv42])
    deconv41 = Deconv2D(256, 3)(concat4)
    deconv42 = Deconv2D(128, 3)(deconv41)
    deconv43 = UpSampling2D()(deconv42)

    padding4 = ZeroPadding2D(((0, 1), (0, 1)))(deconv43)
    concat3 = Concatenate()([padding4, conv32])
    deconv31 = Deconv2D(256, 3)(concat3)
    deconv32 = Deconv2D(128, 3)(deconv31)
    deconv33 = UpSampling2D()(deconv32)

    concat2 = Concatenate()([deconv33, conv22])
    deconv21 = Deconv2D(128, 3)(concat2)
    deconv22 = Deconv2D(64, 3)(deconv21)
    deconv23 = UpSampling2D()(deconv22)

    concat1 = Concatenate()([deconv23, conv12])
    deconv11 = Deconv2D(64, 3)(concat1)
    deconv12 = Deconv2D(32, 3)(deconv11)

    fcn_out = Conv2D(1, 1, activation='sigmoid')(deconv12)

    return Model(inputs=[fcn_in], outputs=[fcn_out])


if __name__ == '__main__':
    net = _build()
    net.load_weights('E:/Data/ShipDetection/FCN/model.h5')

    path = 'E:/Data/ShipDetection/FCN/train'
    imgs = os.listdir(path)[100:120]
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
        plt.imshow(y)
    plt.show()
