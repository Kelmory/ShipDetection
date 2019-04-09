from keras.models import Model
from keras.layers import Input
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras.layers import SeparableConv2D, BatchNormalization
from keras.layers import Concatenate
from keras.layers import GlobalMaxPooling2D, Dense
from keras.layers import GaussianNoise
from keras.optimizers import SGD, Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from ShipDetection.data_io import PathHelper
from ShipDetection.loss import *


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
    _divide_k = 10

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
            callbacks = [EarlyStopping(patience=5, monitor='val_loss'),
                         ModelCheckpoint(self._model_path)]
        else:
            callbacks = kwargs['callbacks']

        if self._model_exist:
            print('[Info]Model weight exists, try loading.')
            try:
                self._net.load_weights(self._model_path)
            except ValueError:
                print('[Info]Model structure does not match, forming new model.')

        valid_steps = int(self._steps / self._divide_k)
        self._net.fit_generator(generator=kwargs['generator'],
                                steps_per_epoch=self._steps - valid_steps,
                                epochs=self._epoch, verbose=1, callbacks=callbacks,
                                validation_data=kwargs['valid_generator'],
                                validation_steps=valid_steps)

    def predict(self, **kwargs):
        self.compile()

        if 'generator' not in kwargs:
            raise KeyError('Batch generator key `generator` not given.')

        if not self._model_exist:
            raise AttributeError('[Info]Model weight does not exist, cannot predict.')

        self._net.load_weights(self._model_path)
        return self._net.predict_generator(generator=kwargs['generator'], steps_per_epoch=self._steps)

    def config_sync(self, config):
        if config is None:
            return
        for setting in config.config:
            if hasattr(self, setting):
                self.__setattr__(setting, config.config[setting])

    def get_model(self) -> Model:
        return self._net

    def __str__(self):
        self._net.summary()
        return ''


class UNet(Net):
    _batch_size = 4
    _x_shape = (768, 768, 1)

    def _build(self):
        fcn_in = Input(shape=self._x_shape)

        # convolution
        gn1 = GaussianNoise(0.1)(fcn_in)
        bn1 = BatchNormalization()(gn1)
        conv11 = Conv2D(64, 3, padding='same', activation='elu')(bn1)
        conv12 = Conv2D(64, 3, padding='same', activation='elu')(conv11)
        conv13 = MaxPooling2D()(conv12)

        bn2 = BatchNormalization()(conv13)
        conv21 = SeparableConv2D(128, 3, padding='same', activation='elu')(bn2)
        conv22 = SeparableConv2D(128, 3, padding='same', activation='elu')(conv21)
        conv23 = MaxPooling2D()(conv22)

        bn3 = BatchNormalization()(conv23)
        conv31 = SeparableConv2D(256, 3, padding='same', activation='elu')(bn3)
        conv32 = SeparableConv2D(256, 3, padding='same', activation='elu')(conv31)
        conv33 = MaxPooling2D()(conv32)

        bn4 = BatchNormalization()(conv33)
        conv41 = SeparableConv2D(512, 3, padding='same', activation='elu')(bn4)
        conv42 = SeparableConv2D(512, 3, padding='same', activation='elu')(conv41)
        conv43 = MaxPooling2D()(conv42)

        bn5 = BatchNormalization()(conv43)
        conv51 = SeparableConv2D(1024, 3, padding='same', activation='elu')(bn5)
        conv52 = SeparableConv2D(1024, 3, padding='same', activation='elu')(conv51)

        # deconvolution
        deconv51 = SeparableConv2D(128, 3, padding='same', activation='elu')(conv52)
        deconv52 = SeparableConv2D(128, 3, padding='same', activation='elu')(deconv51)
        deconv53 = UpSampling2D()(deconv52)

        concat4 = Concatenate()([deconv53, conv42])
        deconv41 = SeparableConv2D(256, 3, padding='same', activation='elu')(concat4)
        deconv42 = SeparableConv2D(128, 3, padding='same', activation='elu')(deconv41)
        deconv43 = UpSampling2D()(deconv42)

        concat3 = Concatenate()([deconv43, conv32])
        deconv31 = SeparableConv2D(256, 3, padding='same', activation='elu')(concat3)
        deconv32 = SeparableConv2D(128, 3, padding='same', activation='elu')(deconv31)
        deconv33 = UpSampling2D()(deconv32)

        concat2 = Concatenate()([deconv33, conv22])
        deconv21 = SeparableConv2D(128, 3, padding='same', activation='elu')(concat2)
        deconv22 = SeparableConv2D(64, 3, padding='same', activation='elu')(deconv21)
        deconv23 = UpSampling2D()(deconv22)

        concat1 = Concatenate()([deconv23, conv12])
        deconv11 = Conv2D(64, 3, padding='same', activation='elu')(concat1)
        deconv12 = Conv2D(32, 3, padding='same', activation='elu')(deconv11)

        fcn_out = Conv2D(1, 1, activation='sigmoid')(deconv12)

        return Model(inputs=[fcn_in], outputs=[fcn_out])

    def compile(self):
        optimizer = Adam(lr=self._lr)
        self._net.compile(optimizer=optimizer, loss=[focal_loss_fixed], metrics=['binary_accuracy'])


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
        self._net.compile(optimizer=optimizer, loss=[loss], metrics=['accuracy'])
