import os
from keras.models import Model, Sequential
from keras.layers import Input
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose
from keras.layers import SeparableConv2D, BatchNormalization
from keras.layers import Concatenate
from keras.layers import GlobalAveragePooling2D, Dense
from keras.layers import GaussianNoise
from keras.applications.mobilenetv2 import MobileNetV2
from keras.layers import LeakyReLU
from keras.optimizers import *
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, LambdaCallback
from ShipDetection.data_io import PathHelper
from ShipDetection.loss import *
from keras.models import load_model
from keras.utils import multi_gpu_model
from keras.initializers import Constant


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
    _gpus = 1

    def __init__(self, config):
        self.config_sync(config)
        self._net = self._build()
        if self._gpus > 1:
            self._multi_net = multi_gpu_model(self._net, gpus=self._gpus)
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

        callbacks = [EarlyStopping(patience=20, monitor='val_loss', verbose=2),
                     ModelCheckpoint(self._model_path, verbose=2, save_best_only=True),
                     CSVLogger(self._model_path[:-3] + '_history.csv', append=self._load_model)]

        if 'callbacks' in kwargs:
            call_type = [type(c) for c in callbacks]
            for call in kwargs['callbacks']:
                if type(call) in call_type:
                    try:
                        callbacks.remove(callbacks[call_type.index(type(call))])
                    except Exception:
                        continue
            callbacks += kwargs['callbacks']

        if self._model_exist and self._load_model:
            print('[Info]Model weight exists, try loading.')
            try:
                self._net.load_weights(self._model_path, by_name=True)
            except (ValueError, OSError):
                print('[Info]Model structure does not match, forming new model.')

        valid_steps = int(self._steps / self._divide_k)
        if self._gpus > 1:
            net = self._multi_net
        else:
            net = self._net
        return net.fit_generator(generator=kwargs['generator'],
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

    def test(self, **kwargs):
        self.compile()

        if 'generator' not in kwargs:
            raise KeyError('Batch generator key `generator` not given.')

        if not self._model_exist:
            raise ValueError('Model weight does not exist, cannot predict.')

        self._net.load_weights(self._model_path)
        return self._net.evaluate_generator(generator=kwargs['generator'], steps=self._steps)

    def summary(self, **kwargs):
        self._net.summary(line_length=120)
        subnets = []
        for i, layer in enumerate(self._net.layers):
            if isinstance(layer, Model):
                print('%-3d\t%-20s\t%-20s\t%-10d\t%-5s'
                      % (i + 1, layer.name, layer.get_output_shape_at(-1), layer.count_params(), layer.trainable))
                print('_' * 120)
                subnets.append(layer)
            else:
                print('%-3d\t%-20s\t%-20s\t%-10d\t%-5s'
                      % (i + 1, layer.name, layer.output_shape, layer.count_params(), layer.trainable))
                print('_' * 120)

        for model in subnets:
            print('\n')
            model.summary(line_length=120)

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
    _x_shape = (None, None, 1)

    def _build(self):
        fcn_in = Input(shape=self._x_shape)

        # convolution
        gn1 = Conv2D(1, 5, kernel_initializer=Constant(0.04), trainable=False, padding='same')(fcn_in)
        bn1 = BatchNormalization()(gn1)
        conv11 = Conv2D(64, 3, padding='same', activation='relu')(bn1)
        conv12 = Conv2D(64, 3, padding='same', activation='relu')(conv11)
        conv13 = MaxPooling2D()(conv12)

        bn2 = BatchNormalization()(conv13)
        conv21 = Conv2D(64, 3, padding='same', activation='relu')(bn2)
        conv22 = Conv2D(64, 3, padding='same', activation='relu')(conv21)
        conv23 = MaxPooling2D()(conv22)

        bn3 = BatchNormalization()(conv23)
        conv31 = Conv2D(128, 3, padding='same', activation='relu')(bn3)
        conv32 = Conv2D(128, 3, padding='same', activation='relu')(conv31)
        conv33 = MaxPooling2D()(conv32)

        bn4 = BatchNormalization()(conv33)
        conv41 = Conv2D(256, 3, padding='same', activation='relu')(bn4)
        conv42 = Conv2D(256, 3, padding='same', activation='relu')(conv41)
        conv43 = MaxPooling2D()(conv42)

        bn5 = BatchNormalization()(conv43)
        conv51 = Conv2D(256, 3, padding='same', activation='relu')(bn5)
        conv52 = Conv2D(256, 3, padding='same', activation='relu')(conv51)

        # deconvolution
        deconv51 = Conv2DTranspose(128, 3, padding='same', activation='relu')(conv52)
        deconv52 = Conv2DTranspose(128, 3, padding='same', activation='relu')(deconv51)
        deconv53 = UpSampling2D()(deconv52)

        concat4 = Concatenate()([deconv53, conv42])
        deconv41 = Conv2DTranspose(128, 3, padding='same', activation='relu')(concat4)
        deconv42 = Conv2DTranspose(128, 3, padding='same', activation='relu')(deconv41)
        deconv43 = UpSampling2D()(deconv42)

        concat3 = Concatenate()([deconv43, conv32])
        deconv31 = Conv2DTranspose(128, 3, padding='same', activation='relu')(concat3)
        deconv32 = Conv2DTranspose(128, 3, padding='same', activation='relu')(deconv31)
        deconv33 = UpSampling2D()(deconv32)

        concat2 = Concatenate()([deconv33, conv22])
        deconv21 = Conv2DTranspose(64, 3, padding='same', activation='relu')(concat2)
        deconv22 = Conv2DTranspose(64, 3, padding='same', activation='relu')(deconv21)
        deconv23 = UpSampling2D()(deconv22)

        concat1 = Concatenate()([deconv23, conv12])
        deconv11 = Conv2DTranspose(64, 3, padding='same', activation='relu')(concat1)
        deconv12 = Conv2DTranspose(32, 3, padding='same', activation='relu')(deconv11)

        channel_out = Conv2D(16, 1, activation='relu')(deconv12)
        fcn_out = Conv2D(1, 1, activation='hard_sigmoid')(channel_out)
        return Model(inputs=[fcn_in], outputs=[fcn_out])

    def compile(self):
        optimizer = Adam(lr=self._lr)
        self._net.compile(optimizer=optimizer, loss=[focal_loss(alpha=0.75, gamma=2)],
                          metrics=[non_zero_rate, true_non_zero_rate, true_positive_rate])
        if self._gpus > 1:
            self._multi_net.compile(optimizer=optimizer, loss=[focal_loss(alpha=0.75, gamma=2)],
                                    metrics=['binary_accuracy', true_positive_rate])

    def advanced_train(self, **kwargs):
        mod_callbacks = [ModelCheckpoint(self._model_path, monitor='val_true_positive_rate',
                                     verbose=2, save_best_only=True)]
        self.train(callbacks=mod_callbacks, **kwargs)


class ConvNN(Net):
    _epoch = 200
    _batch_size = 16
    _x_shape = (None, None, 1)
    _lr = 1e-4

    def _build(self):
        cnn_in = Input(shape=self._x_shape)

        def conv_block(input_, filters=16, kernel_size=3):
            # input_ = BatchNormalization()(input_)
            conv1 = SeparableConv2D(filters, kernel_size, padding='same', activation='relu')(input_)
            conv2 = SeparableConv2D(filters, kernel_size, padding='same',
                                    activation='relu', dilation_rate=(2, 1))(input_)
            conv3 = SeparableConv2D(filters, kernel_size, padding='same',
                                    activation='relu', dilation_rate=(1, 2))(input_)
            conv4 = SeparableConv2D(filters, kernel_size, padding='same', activation='relu')(conv1)
            concat = Concatenate()([conv1, conv2, conv3, conv4])
            conv = Conv2D(filters, kernel_size, padding='same', activation='hard_sigmoid')(concat)
            return conv

        def conv_down_block(input_, filters=16, kernel_size=3):
            # input_ = BatchNormalization()(input_)
            conv1 = SeparableConv2D(filters, kernel_size, padding='same', activation='relu')(input_)
            conv2 = SeparableConv2D(filters, kernel_size, padding='same', activation='relu')(conv1)
            concat = Concatenate()([conv1, conv2])
            conv = Conv2D(filters, kernel_size, padding='same', activation='hard_sigmoid')(concat)
            return MaxPooling2D()(conv)

        conv1 = conv_block(cnn_in, 16)
        concat = Concatenate()([cnn_in, conv1])

        conv2 = conv_down_block(concat, 32)
        conv3 = conv_block(conv2, 32)
        concat = Concatenate()([conv2, conv3])

        conv4 = conv_down_block(concat, 64)
        conv5 = conv_block(conv4, 64)
        concat = Concatenate()([conv4, conv5])

        conv6 = conv_down_block(concat, 128)
        conv7 = conv_block(conv6, 128)
        concat = Concatenate()([conv6, conv7])

        # global pooling
        gap = GlobalAveragePooling2D()(concat)
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

            conv1 = SeparableConv2D(filters, kernel_size, padding='same', kernel_initializer='RandomUniform')(input_)
            conv1 = LeakyReLU()(conv1)
            conv1 = SeparableConv2D(filters, kernel_size, padding='same', kernel_initializer='RandomUniform')(conv1)
            conv1 = LeakyReLU()(conv1)

            conv2 = SeparableConv2D(filters, kernel_size, padding='same', kernel_initializer='RandomUniform')(input_)
            conv2 = LeakyReLU()(conv2)

            concat = Concatenate()([conv1, conv2])
            concat = SeparableConv2D(filters * 2, kernel_size, padding='same', activation='relu')(concat)
            concat = Conv2D(filters, kernel_size, padding='same', activation='hard_sigmoid')(concat)

            return MaxPooling2D()(concat)

        def up_sample_block(input_, filters=32, kernel_size=3):
            # input_ = BatchNormalization()(input_)

            conv1 = SeparableConv2D(filters, kernel_size, padding='same', kernel_initializer='RandomUniform')(input_)
            conv1 = LeakyReLU()(conv1)

            conv2 = SeparableConv2D(filters, kernel_size, padding='same', activation='relu')(input_)
            conv2 = SeparableConv2D(filters, kernel_size, padding='same', activation='relu')(conv2)

            concat = Concatenate()([conv1, conv2])

            conv3 = SeparableConv2D(filters, kernel_size, padding='same', activation='hard_sigmoid')(concat)

            return UpSampling2D()(conv3)

        net_in = Input(self._x_shape)
        db1 = down_sample_block(net_in, 16)
        db2 = down_sample_block(db1, 16)
        db3 = down_sample_block(db2, 16)
        db4 = down_sample_block(db3, 32)
        db5 = down_sample_block(db4, 32)
        db6 = down_sample_block(db5, 64)
        db7 = down_sample_block(db6, 64)

        ub7 = up_sample_block(db7, 128)
        concat = Concatenate()([ub7, db6])
        ub6 = up_sample_block(concat, 128)
        concat = Concatenate()([ub6, db5])
        ub5 = up_sample_block(concat, 64)
        concat = Concatenate()([ub5, db4])
        ub4 = up_sample_block(concat, 64)
        concat = Concatenate()([ub4, db3])
        ub3 = up_sample_block(concat, 32)
        concat = Concatenate()([ub3, db2])
        ub2 = up_sample_block(concat, 32)
        concat = Concatenate()([ub2, db1])
        ub1 = up_sample_block(concat, 32)

        class_out = SeparableConv2D(128, 1, activation='relu')(ub1)
        net_out = Conv2D(1, 1, activation='hard_sigmoid')(class_out)
        return Model(inputs=[net_in], outputs=[net_out])

    def compile(self):
        optimizer = RMSprop(self._lr)
        self._net.compile(optimizer, loss=[focal_loss(alpha=0.95, gamma=2)],
                          metrics=['binary_crossentropy', non_zero_rate, true_positive_rate])
        if self._gpus > 1:
            self._multi_net.compile(optimizer=optimizer, loss=[focal_loss(alpha=0.95, gamma=0.5)],
                                    metrics=['binary_accuracy', non_zero_rate, true_positive_rate])


class TransferUNet(Net):
    def _build(self):
        subnet = MobileNetV2(include_top=False)
        subnet.trainable = False

        net_in = Input(self._x_shape)
        down_layer = MaxPooling2D()(net_in)
        fixed_layer = Conv2D(3, 1)(down_layer)
        fixed = subnet(fixed_layer)

        net_out = Conv2D(1, 1, activation='hard_sigmoid')(fixed)

        return Model(inputs=[net_in], outputs=[net_out])

    def compile(self):
        optimizer = Adam(self._lr)
        self._net.compile(optimizer, loss=[focal_loss()],
                          metrics=[true_negative_rate, true_positive_rate, non_zero_rate])


class LoadNet(Net):
    def _build(self) -> Model:
        return load_model(self._model_path, compile=False)

    def compile(self):
        optimizer = Adam(lr=self._lr)
        self._net.compile(optimizer=optimizer, loss=[focal_loss(alpha=0.75, gamma=2)],
                          metrics=[non_zero_rate, true_non_zero_rate, true_positive_rate])


class SegNet(Net):
    def _build(self) -> Model:
        fcn_in = Input(shape=self._x_shape)

        # convolution
        gn1 = Conv2D(1, 5, kernel_initializer=Constant(0.04), trainable=False, padding='same')(fcn_in)
        bn1 = BatchNormalization()(gn1)
        conv11 = Conv2D(64, 3, padding='same', activation='relu')(bn1)
        conv12 = Conv2D(64, 3, padding='same', activation='relu')(conv11)
        conv13 = MaxPooling2D()(conv12)

        bn2 = BatchNormalization()(conv13)
        conv21 = SeparableConv2D(64, 3, padding='same', activation='relu')(bn2)
        conv22 = SeparableConv2D(64, 3, padding='same', activation='relu')(conv21)
        conv23 = MaxPooling2D()(conv22)

        bn3 = BatchNormalization()(conv23)
        conv31 = SeparableConv2D(128, 3, padding='same', activation='relu')(bn3)
        conv32 = SeparableConv2D(128, 3, padding='same', activation='relu')(conv31)
        conv33 = MaxPooling2D()(conv32)

        bn4 = BatchNormalization()(conv33)
        conv41 = SeparableConv2D(256, 3, padding='same', activation='relu')(bn4)
        conv42 = SeparableConv2D(256, 3, padding='same', activation='relu')(conv41)
        conv43 = MaxPooling2D()(conv42)

        bn5 = BatchNormalization()(conv43)
        conv51 = SeparableConv2D(256, 3, padding='same', activation='relu')(bn5)
        conv52 = SeparableConv2D(256, 3, padding='same', activation='relu')(conv51)

        # deconvolution
        deconv51 = SeparableConv2D(128, 3, padding='same', activation='relu')(conv52)
        deconv52 = SeparableConv2D(128, 3, padding='same', activation='relu')(deconv51)
        deconv53 = UpSampling2D()(deconv52)

        deconv41 = SeparableConv2D(128, 3, padding='same', activation='relu')(deconv53)
        deconv42 = SeparableConv2D(128, 3, padding='same', activation='relu')(deconv41)
        deconv43 = UpSampling2D()(deconv42)

        deconv31 = SeparableConv2D(128, 3, padding='same', activation='relu')(deconv43)
        deconv32 = SeparableConv2D(128, 3, padding='same', activation='relu')(deconv31)
        deconv33 = UpSampling2D()(deconv32)

        deconv21 = SeparableConv2D(64, 3, padding='same', activation='relu')(deconv33)
        deconv22 = SeparableConv2D(64, 3, padding='same', activation='relu')(deconv21)
        deconv23 = UpSampling2D()(deconv22)

        deconv11 = Conv2D(64, 3, padding='same', activation='relu')(deconv23)
        deconv12 = Conv2D(32, 3, padding='same', activation='relu')(deconv11)

        channel_out = Conv2D(16, 1, activation='relu')(deconv12)
        fcn_out = Conv2D(1, 1, activation='hard_sigmoid')(channel_out)
        return Model(inputs=[fcn_in], outputs=[fcn_out])

    def compile(self):
        optimizer = Adam(self._lr)
        self._net.compile(optimizer, loss=[focal_loss(0, 0.99)],
                          metrics=['binary_accuracy', true_positive_rate])


class ConvNN2(Net):
    def _build(self):
        cnn_in = Input(shape=self._x_shape)
        bn = BatchNormalization()(cnn_in)

        def conv_block(input_, filters=16, kernel_size=3):
            # input_ = BatchNormalization()(input_)
            conv1 = SeparableConv2D(filters, kernel_size, padding='same', activation='elu')(input_)
            conv4 = SeparableConv2D(filters, kernel_size, padding='same', activation='relu')(conv1)
            concat = Concatenate()([conv1, conv4])
            conv = Conv2D(filters, kernel_size, padding='same', activation='hard_sigmoid')(concat)
            return MaxPooling2D()(conv)

        conv1 = conv_block(bn, 32)
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

