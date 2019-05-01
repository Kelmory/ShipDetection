from keras.models import load_model, save_model
from keras import Model
from keras.layers import Input, BatchNormalization, GaussianNoise, Conv2D
from keras.layers import SeparableConv2D, MaxPooling2D, UpSampling2D, Concatenate


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


if __name__ == '__main__':
    net = build((None, None, 1))
    origin = build((768, 768, 1))
    origin.load_weights('E:/Data/ShipDetection/FCN/UNet_768_epoch20.h5', reshape=True)
    net.set_weights(origin.get_weights())
    save_model(net, 'E:/Data/ShipDetection/FCN/UNet_Any_epoch20.h5')
    net.summary(line_length=120)
