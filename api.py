from keras import Model
from keras.models import load_model
import numpy as np
from ShipDetection.data_io import PathHelper


class SyntheticPipeline:
    _roi_net: Model
    _cls_net: Model

    def __init__(self, roi_path, cls_path):
        if not PathHelper(roi_path).is_file_valid():
            raise ValueError
        if not PathHelper(cls_path).is_file_valid():
            raise ValueError
        self._roi_net = load_model(roi_path, compile=False)
        self._cls_net = load_model(cls_path, compile=False)

    def end2end_predict(self, img: np.ndarray) -> ([(int, int)], np.ndarray):
        return

    def _normalize_img(self, img: np.ndarray) -> np.ndarray:
        return

