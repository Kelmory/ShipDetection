from keras.models import Model
from keras.layers import Input
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras.layers import SeparableConv2D, BatchNormalization
from keras.layers import Concatenate
from keras.layers import GlobalAveragePooling2D, Dense
from keras.layers import GaussianNoise
from keras.layers import LeakyReLU
from keras.optimizers import *
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
    _load_model = True
    _load = False
    _steps = 1
    _divide_k = 10
    _trainable = (None, None)  # (start, end)

    def __init__(self, config):
        self.config_sync(config)
        self._net = self._build()
        self._set_trainable()
        self._model_exist = PathHelper(self._model_path).is_file_valid()

    def _build(self) -> Model:
        pass

    def compile(self):
        pass

    def _set_trainable(self):
        start, end = self._trainable
        if start is None and end is None:
            return
        else:
            for layer in self._net.layers:
                layer.trainable = False
            for layer in self._net.layers[start:end]:
                layer.trainable = True

    def train(self, **kwargs):
        self.compile()

        if 'generator' not in kwargs:
            raise KeyError('Batch generator key `generator` not given.')

        if 'callbacks' not in kwargs:
            callbacks = [EarlyStopping(patience=50, monitor='val_loss'),
                         ModelCheckpoint(self._model_path, save_best_only=True)]
        else:
            callbacks = kwargs['callbacks']

        if self._model_exist and self._load_model:
            print('[Info]Model weight exists, try loading.')
            try:
                self._net.load_weights(self._model_path, by_name=True)
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
            raise ValueError('Model weight does not exist, cannot predict.')

        self._net.load_weights(self._model_path)
        return self._net.predict_generator(generator=kwargs['generator'], steps_per_epoch=self._steps)

    def summary(self, **kwargs):
        self._net.summary()
        for i, layer in enumerate(self._net.layers):
            print('%-3d\t%-20s\t%-20s\t%-5s' % (i + 1, layer.name, layer.output_shape, layer.trainable))
            print('_' * 80)

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
        conv11 = Conv2D(64, 3, padding='same', activation='relu')(bn1)
        conv12 = Conv2D(64, 3, padding='same', activation='relu')(conv11)
        conv13 = MaxPooling2D()(conv12)

        bn2 = BatchNormalization()(conv13)
        conv21 = SeparableConv2D(128, 3, padding='same', activation='relu')(bn2)
        conv22 = SeparableConv2D(128, 3, padding='same', activation='relu')(conv21)
        conv23 = MaxPooling2D()(conv22)

        bn3 = BatchNormalization()(conv23)
        conv31 = SeparableConv2D(256, 3, padding='same', activation='relu')(bn3)
        conv32 = SeparableConv2D(256, 3, padding='same', activation='relu')(conv31)
        conv33 = MaxPooling2D()(conv32)

        bn4 = BatchNormalization()(conv33)
        conv41 = SeparableConv2D(512, 3, padding='same', activation='relu')(bn4)
        conv42 = SeparableConv2D(512, 3, padding='same', activation='relu')(conv41)
        conv43 = MaxPooling2D()(conv42)

        bn5 = BatchNormalization()(conv43)
        conv51 = SeparableConv2D(1024, 3, padding='same', activation='relu')(bn5)
        conv52 = SeparableConv2D(1024, 3, padding='same', activation='relu')(conv51)

        # deconvolution
        deconv51 = SeparableConv2D(128, 3, padding='same', activation='relu')(conv52)
        deconv52 = SeparableConv2D(128, 3, padding='same', activation='relu')(deconv51)
        deconv53 = UpSampling2D()(deconv52)

        concat4 = Concatenate()([deconv53, conv42])
        deconv41 = SeparableConv2D(256, 3, padding='same', activation='relu')(concat4)
        deconv42 = SeparableConv2D(128, 3, padding='same', activation='relu')(deconv41)
        deconv43 = UpSampling2D()(deconv42)

        concat3 = Concatenate()([deconv43, conv32])
        deconv31 = SeparableConv2D(256, 3, padding='same', activation='relu')(concat3)
        deconv32 = SeparableConv2D(128, 3, padding='same', activation='relu')(deconv31)
        deconv33 = UpSampling2D()(deconv32)

        concat2 = Concatenate()([deconv33, conv22])
        deconv21 = SeparableConv2D(128, 3, padding='same', activation='relu')(concat2)
        deconv22 = SeparableConv2D(64, 3, padding='same', activation='relu')(deconv21)
        deconv23 = UpSampling2D()(deconv22)

        concat1 = Concatenate()([deconv23, conv12])
        deconv11 = Conv2D(64, 3, padding='same', activation='relu')(concat1)
        deconv12 = Conv2D(32, 3, padding='same', activation='relu')(deconv11)

        fcn_out = Conv2D(1, 1, activation='sigmoid')(deconv12)
        return Model(inputs=[fcn_in], outputs=[fcn_out])

    def compile(self):
        optimizer = Adadelta(lr=self._lr)
        self._net.compile(optimizer=optimizer, loss=[focal_loss_fixed], metrics=['binary_accuracy'])


class ConvNN(Net):
    _epoch = 200
    _batch_size = 16
    _x_shape = (None, None, 1)
    _lr = 1e-4

    def _build(self):
        cnn_in = Input(shape=self._x_shape)
        bn = BatchNormalization()(cnn_in)

        def conv_block(input_, filters=16, kernel_size=3):
            # input_ = BatchNormalization()(input_)
            conv1 = SeparableConv2D(filters, kernel_size, padding='same', activation='relu')(input_)
            conv2 = SeparableConv2D(filters, kernel_size, padding='same',
                                    activation='relu', dilation_rate=(2, 1))(input_)
            conv3 = SeparableConv2D(filters, kernel_size, padding='same',
                                    activation='relu', dilation_rate=(1, 2))(input_)
            concat = Concatenate()([conv1, conv2, conv3])
            conv = Conv2D(filters, kernel_size,padding='same', activation='hard_sigmoid')(concat)
            return MaxPooling2D()(conv)

        conv1 = conv_block(bn)
        conv2 = conv_block(conv1, 32)
        conv3 = conv_block(conv2, 64)
        conv4 = conv_block(conv3, 128)
        conv5 = conv_block(conv4, 256)

        # global pooling
        gap = GlobalAveragePooling2D()(conv5)
        cnn_out = Dense(1, activation='sigmoid')(gap)

        return Model(inputs=[cnn_in], outputs=[cnn_out])

    def compile(self):
        optimizer = Adam(lr=self._lr)
        loss = 'binary_crossentropy'
        self._net.compile(optimizer=optimizer, loss=[loss], metrics=['accuracy'])


class DenoiseNet(Net):
    _x_shape = (None, None, 1)
    _y_shape = _x_shape

    def _build(self):
        def down_sample_block(input_, filters=32, kernel_size=3):
            # input_ = BatchNormalization()(input_)

            conv1 = SeparableConv2D(filters, kernel_size, padding='same')(input_)
            conv1 = LeakyReLU()(conv1)
            conv1 = SeparableConv2D(filters, kernel_size, padding='same')(conv1)
            conv1 = LeakyReLU()(conv1)

            conv2 = SeparableConv2D(filters, kernel_size, dilation_rate=(2, 1), padding='same')(input_)
            conv2 = LeakyReLU()(conv2)

            conv3 = SeparableConv2D(filters, kernel_size, dilation_rate=(1, 2), padding='same')(input_)
            conv3 = LeakyReLU()(conv3)

            conv4 = SeparableConv2D(filters, kernel_size, padding='same')(input_)
            conv4 = LeakyReLU()(conv4)

            concat = Concatenate()([conv1, conv2, conv3, conv4])
            concat = Conv2D(filters * 4, kernel_size, padding='same', activation='relu')(concat)
            concat = Conv2D(filters * 2, kernel_size, padding='same', activation='hard_sigmoid')(concat)

            return MaxPooling2D()(concat)

        def up_sample_block(input_, filters=32, kernel_size=3):
            # input_ = BatchNormalization()(input_)

            conv1 = SeparableConv2D(filters, kernel_size, padding='same')(input_)
            conv1 = LeakyReLU()(conv1)

            conv2 = SeparableConv2D(filters, kernel_size, padding='same')(conv1)
            conv2 = LeakyReLU()(conv2)

            conv3 = Conv2D(filters, kernel_size, padding='same', activation='hard_sigmoid')(conv2)

            return UpSampling2D()(conv3)

        net_in = Input(self._x_shape)
        db1 = down_sample_block(net_in)
        db2 = down_sample_block(db1)
        db3 = down_sample_block(db2, 64)
        db4 = down_sample_block(db3, 64)
        db5 = down_sample_block(db4, 128)
        db6 = down_sample_block(db5, 128)
        db7 = down_sample_block(db6, 256)

        ub7 = up_sample_block(db7, 256)
        concat = Concatenate()([ub7, db6])
        ub6 = up_sample_block(concat, 128)
        concat = Concatenate()([ub6, db5])
        ub5 = up_sample_block(concat, 128)
        concat = Concatenate()([ub5, db4])
        ub4 = up_sample_block(concat, 64)
        concat = Concatenate()([ub4, db3])
        ub3 = up_sample_block(concat, 64)
        concat = Concatenate()([ub3, db2])
        ub2 = up_sample_block(concat)
        concat = Concatenate()([ub2, db1])
        ub1 = up_sample_block(concat)

        class_out = Conv2D(2, 1, activation='relu')(ub1)
        net_out = Conv2D(1, 1, activation='hard_sigmoid')(class_out)
        return Model(inputs=[net_in], outputs=[net_out])

    def compile(self):
        optimizer = Adam(self._lr)
        self._net.compile(optimizer, loss=[focal_loss_fixed],
                          metrics=[true_negative_rate, true_positive_rate])
