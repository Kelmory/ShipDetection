from keras.models import Model
from keras.layers import Input
from keras.layers import Conv2D, Deconv2D, MaxPooling2D, UpSampling2D
from keras.layers import SeparableConv2D, BatchNormalization
from keras.layers import Concatenate, ZeroPadding2D
from keras.layers import GlobalMaxPooling2D, Dense
from keras.optimizers import SGD, Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from ShipDetection.data_io import PathHelper
import keras.backend as K


def iou_metric(y_true, y_pred):
    union = K.maximum(y_true, y_pred)
    intersection = K.minimum(y_true, y_pred)
    return K.sum(intersection) / K.sum(union)


class Net:
    _net = None
    _epoch = 100
    _batch_size = 8
    _x_shape = (None, None, 1)
    _y_shape = (None, 1)
    _lr = 1e-4
    _model_path = ''
    _model_exist = False
    _load = False
    _steps = 1

    def __init__(self, config):
        self.config_sync(config)
        self._net = self._build()
        self._model_exist = PathHelper(self._model_path).is_file_valid()

    def _build(self):
        pass

    def compile(self):
        pass

    def train(self, **kwargs):
        self.compile()

        if 'generator' not in kwargs:
            raise KeyError('Batch generator key `generator` not given.')
        if 'callbacks' not in kwargs:
            callbacks = [EarlyStopping(patience=5, monitor='loss'),
                         ModelCheckpoint(self._model_path)]
        else:
            callbacks = kwargs['callbacks']

        if self._model_exist:
            print('[Info]Model weight exists, loading.')
            self._net.load_weights(self._model_path)

        self._net.fit_generator(generator=kwargs['generator'], steps_per_epoch=self._steps,
                                epochs=self._epoch, verbose=1, callbacks=callbacks)

    def predict(self, **kwargs):
        self.compile()

        if 'generator' not in kwargs:
            raise KeyError('Batch generator key `generator` not given.')

        if not self._model_exist:
            raise AttributeError('[Info]Model weight does not exist, cannot predict.')

        self._net.load_weights(self._model_path)
        return self._net.predict_generator(generator=kwargs['generator'], steps_per_epoch=self._steps)

    def config_sync(self, config):
        for setting in config.config:
            if hasattr(self, setting):
                self.__setattr__(setting, config.config[setting])

    def __str__(self):
        self._net.summary()
        return ''


class UNet(Net):
    _batch_size = 4
    _x_shape = (768, 768, 1)

    def _build(self):
        fcn_in = Input(shape=self._x_shape)

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

    def compile(self):
        optimizer = Adam()
        loss = 'binary_crossentropy'
        self._net.compile(optimizer=optimizer, loss=loss, metrics=[iou_metric])


class ConvNN(Net):
    _epoch = 200
    _batch_size = 16
    _x_shape = (None, None, 1)
    _lr = 1e-4

    def _build(self):
        cnn_in = Input(shape=self._x_shape)
        bn = BatchNormalization()(cnn_in)
        # separable convolution 1
        conv1 = SeparableConv2D(64, kernel_size=5, activation='relu')(bn)
        # separable convolution 2
        conv2 = SeparableConv2D(64, kernel_size=5, activation='relu')(conv1)
        # separable convolution 3
        conv3 = SeparableConv2D(128, kernel_size=3, strides=(3, 3), activation='relu')(conv2)
        # separable convolution 4
        conv4 = SeparableConv2D(256, kernel_size=3, strides=(3, 3), activation='relu')(conv3)
        # separable convolution 5
        conv5 = SeparableConv2D(512, kernel_size=3, activation='relu')(conv4)
        # global pooling
        gap = GlobalMaxPooling2D()(conv5)
        cnn_out = Dense(1, activation='sigmoid')(gap)

        return Model(inputs=[cnn_in], outputs=[cnn_out])

    def compile(self):
        optimizer = SGD(lr=self._lr)
        loss = 'binary_crossentropy'
        self._net.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])


