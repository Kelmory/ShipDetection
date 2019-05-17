from keras.models import load_model, save_model
from keras import Model
from keras.layers import Input, BatchNormalization, GaussianNoise, Conv2D
from keras.layers import SeparableConv2D, MaxPooling2D, UpSampling2D, Concatenate
from keras.layers import LeakyReLU
import matplotlib.pyplot as plt
import cv2


def build(shape):
    net_in = Input(shape)
    bn1 = BatchNormalization()(net_in)
    gn1 = GaussianNoise(0.1)(bn1)

    conv11 = SeparableConv2D(64, 3, padding='same', activation='relu')(gn1)
    conv12 = SeparableConv2D(64, 3, padding='same', activation='relu')(conv11)
    pool1 = MaxPooling2D()(conv12)

    conv21 = SeparableConv2D(128, 3, padding='same', activation='relu')(pool1)
    conv22 = SeparableConv2D(128, 3, padding='same', activation='relu')(conv21)
    pool2 = MaxPooling2D()(conv22)

    conv31 = SeparableConv2D(256, 3, padding='same', activation='relu')(pool2)
    conv32 = SeparableConv2D(256, 3, padding='same', activation='relu')(conv31)
    pool3 = MaxPooling2D()(conv32)

    conv41 = SeparableConv2D(512, 3, padding='same', activation='relu')(pool3)
    conv42 = SeparableConv2D(512, 3, padding='same', activation='relu')(conv41)
    pool4 = MaxPooling2D()(conv42)

    conv51 = SeparableConv2D(1024, 3, padding='same', activation='relu')(pool4)
    conv52 = SeparableConv2D(1024, 3, padding='same', activation='relu')(conv51)

    deconv51 = SeparableConv2D(128, 3, padding='same', activation='relu')(conv52)
    deconv52 = SeparableConv2D(128, 3, padding='same', activation='relu')(deconv51)
    up5 = UpSampling2D()(deconv52)

    concat4 = Concatenate()([up5, conv42])
    deconv41 = SeparableConv2D(256, 3, padding='same', activation='relu')(concat4)
    deconv42 = SeparableConv2D(128, 3, padding='same', activation='relu')(deconv41)
    up4 = UpSampling2D()(deconv42)

    concat3 = Concatenate()([up4, conv32])
    deconv31 = SeparableConv2D(256, 3, padding='same', activation='relu')(concat3)
    deconv32 = SeparableConv2D(128, 3, padding='same', activation='relu')(deconv31)
    up3 = UpSampling2D()(deconv32)

    concat2 = Concatenate()([up3, conv22])
    deconv21 = SeparableConv2D(128, 3, padding='same', activation='relu')(concat2)
    deconv22 = SeparableConv2D(64, 3, padding='same', activation='relu')(deconv21)
    up2 = UpSampling2D()(deconv22)

    concat1 = Concatenate()([up2, conv12])
    deconv11 = SeparableConv2D(64, 3, padding='same', activation='relu')(concat1)
    deconv12 = SeparableConv2D(32, 3, padding='same', activation='relu')(deconv11)

    net_out = Conv2D(1, 1, padding='same', activation='sigmoid')(deconv12)

    return Model(inputs=[net_in], outputs=[net_out])


def transfer_with_same_weight():
    net = build((None, None, 1))

    origin = load_model('E:/Data/ShipDetection/FCN/UNet_768_epoch20.h5', compile=False)
    net.load_weights('E:/Data/ShipDetection/FCN/UNet_768_epoch20.h5', by_name=True)

    x = cv2.imread('E:/Data/ShipDetection/FCN/samples/00e90efc3.jpg', 0)
    x = x.reshape((1, 768, 768, 1))

    y1 = net.predict(x).reshape((768, 768))
    y2 = origin.predict(x).reshape((768, 768))

    plt.subplot('121', title='transfer')
    plt.imshow(y1)
    plt.subplot('122', title='origin')
    plt.imshow(y2)
    plt.show()

    # save_model(net, 'E:/Data/ShipDetection/FCN/UNet_Any_epoch20.h5')
    net.summary(line_length=120)


def transfer_from_multi_gpu_model():
    net = load_model('E:/Data/ShipDetection/FCN/UNet_384.h5', compile=False)
    for layer in net.layers:
        if 'model' in layer.name:
            save_model(layer, 'E:/Data/ShipDetection/FCN/UNet_384_Sole.h5')
            break


def transfer_without_batchnorm():
    def _build():
        def down_sample_block(input_, filters=32, kernel_size=3):
            # input_ = BatchNormalization()(input_)

            conv1 = SeparableConv2D(filters, kernel_size, padding='same', kernel_initializer='Ones')(input_)
            conv1 = LeakyReLU()(conv1)
            conv1 = SeparableConv2D(filters, kernel_size, padding='same', kernel_initializer='Ones')(conv1)
            conv1 = LeakyReLU()(conv1)

            conv2 = SeparableConv2D(filters, kernel_size, padding='same', kernel_initializer='Ones')(input_)
            conv2 = LeakyReLU()(conv2)

            concat = Concatenate()([conv1, conv2])
            concat = SeparableConv2D(filters * 2, kernel_size, padding='same', activation='relu')(concat)
            concat = Conv2D(filters, kernel_size, padding='same', activation='hard_sigmoid')(concat)

            return MaxPooling2D()(concat)

        def up_sample_block(input_, filters=32, kernel_size=3):
            # input_ = BatchNormalization()(input_)

            conv1 = SeparableConv2D(filters, kernel_size, padding='same', kernel_initializer='Ones')(input_)
            conv1 = LeakyReLU()(conv1)

            conv3 = SeparableConv2D(filters, kernel_size, padding='same', activation='hard_sigmoid')(conv1)

            return UpSampling2D()(conv3)

        net_in = Input((384, 384, 1))
        db1 = down_sample_block(net_in)
        db2 = down_sample_block(db1)
        db3 = down_sample_block(db2)
        db4 = down_sample_block(db3, 64)
        db5 = down_sample_block(db4, 64)
        db6 = down_sample_block(db5, 128)
        db7 = down_sample_block(db6, 128)

        ub7 = up_sample_block(db7, 128)
        concat = Concatenate()([ub7, db6])
        ub6 = up_sample_block(concat, 128)
        concat = Concatenate()([ub6, db5])
        ub5 = up_sample_block(concat, 64)
        concat = Concatenate()([ub5, db4])
        ub4 = up_sample_block(concat, 64)
        concat = Concatenate()([ub4, db3])
        ub3 = up_sample_block(concat)
        concat = Concatenate()([ub3, db2])
        ub2 = up_sample_block(concat)
        concat = Concatenate()([ub2, db1])
        ub1 = up_sample_block(concat)

        class_out = SeparableConv2D(64, 1, activation='relu')(ub1)
        net_out = Conv2D(1, 1, activation='hard_sigmoid')(class_out)
        return Model(inputs=[net_in], outputs=[net_out])

    net = _build()
    net.load_weights('E:/Data/ShipDetection/FCN/UNet_384_Sole.h5', by_name=True)
    save_model(net, 'E:/Data/ShipDetection/FCN/UNet_384_NonBN.h5')


if __name__ == '__main__':
    transfer_without_batchnorm()
