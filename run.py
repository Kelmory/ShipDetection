import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))


def run_fcn():
    fcn_config_dict = {'_epoch': 100,
                       '_batch_size': 1,
                       '_steps': 2000,
                       '_dropna': True,
                       '_load_model': True,
                       '_lr': 1e-2,
                       '_model_path': 'E:/Data/ShipDetection/FCN/UNet_Any_epoch10.h5',
                       '_parent_path': 'E:/Data/ShipDetection/FCN',
                       '_csv_path': 'train2.csv',
                       '_dir_path': 'train',
                       '_load_mode': 'disk',
                       '_use_mode': 'train',
                       '_divide_k': 5,
                       '_x_shape': (None, None, 1),
                       '_y_shape': (768, 768, 1),
                       '_trainable': (None, None),
                       '_augmentation': False,
                       '_gpus': 1,
                       }
    pipe = Pipeline('UNet', 'CsvDirGenerator', fcn_config_dict)
    pipe.work('train')


def run_fcn2():
    fcn_config_dict = {'_epoch': 20,
                       '_batch_size': 1,
                       '_lr': 1e-4,
                       '_model_path': 'E:/Data/ShipDetection/FCN/model.h5',
                       '_parent_path': 'E:/Data/ShipDetection/FCN',
                       '_csv_path': 'train.csv',
                       '_dir_path': 'train',
                       '_load_mode': 'disk',
                       '_use_mode': 'test',
                       '_steps': 20,
                       '_divide_k': 10,
                       '_x_shape': (256, 256, 1),
                       '_y_shape': (256, 256, 1)
                       }
    pipe = Pipeline('DenoiseNet', 'CsvDirGenerator', fcn_config_dict)
    pipe.work('train')


def run_cnn():
    cnn_config_dict = {'_epoch': 100,
                       '_batch_size': 1,
                       '_lr': 1e-4,
                       '_model_path': 'E:/Data/ShipDetection/CNN/model.h5',
                       '_steps': 9600,
                       '_parent_path': 'E:/Data/ShipDetection/CNN',
                       '_pos_path': 'ship',
                       '_neg_path': 'negative',
                       '_load_mode': 'disk',
                       '_use_mode': 'train',
                       '_divide_k': 10,
                       '_x_shape': (None, None, 1),
                       '_y_shape': (None, 1)
                       }
    pipe = Pipeline('ConvNN', 'BinDirDataGenerator', cnn_config_dict)
    pipe.work('train')


if __name__ == '__main__':
    from ShipDetection.core import Pipeline

    run_fcn()
    # run_cnn()
