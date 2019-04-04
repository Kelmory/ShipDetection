from ShipDetection.core import Pipeline


def run_fcn():
    fcn_config_dict = {'_epoch': 20,
                       '_batch_size': 1,
                       '_lr': 3e-4,
                       '_model_path': 'E:/Data/ShipDetection/FCN/model.h5',
                       '_parent_path': 'E:/Data/ShipDetection/FCN',
                       '_csv_path': 'train2.csv',
                       '_dir_path': 'train',
                       '_load_mode': 'disk',
                       '_use_mode': 'train',
                       '_divide_k': 10,
                       '_x_shape': (768, 768, 1),
                       '_y_shape': (768, 768, 1)
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
                       '_x_shape': (768, 768, 1),
                       '_y_shape': (768, 768, 1)
                       }
    pipe = Pipeline('UNet', 'CsvDirGenerator', fcn_config_dict)
    pipe.work('train')


def run_cnn():
    cnn_config_dict = {'_epoch': 100,
                       '_batch_size': 32,
                       '_lr': 1e-4,
                       '_model_path': 'E:/Data/ShipDetection/CNN/model.h5',
                       '_steps': 3812,
                       '_parent_path': 'E:/Data/ShipDetection/CNN',
                       '_pos_path': 'ship',
                       '_neg_path': 'negative',
                       '_load_mode': 'disk',
                       '_use_mode': 'train',
                       '_divide_k': 10,
                       '_x_shape': (200, 200, 1),
                       '_y_shape': (None, 1)
                       }
    pipe = Pipeline('ConvNN', 'BinDirDataGenerator', cnn_config_dict)
    pipe.work('train')


if __name__ == '__main__':
    run_fcn()
