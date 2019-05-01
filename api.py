from keras import Model
from keras.models import load_model
import cv2
import math
import numpy as np
import skimage.measure as measure
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

    def end2end_predict(self, img: np.ndarray) -> ([tuple], np.ndarray):
        seg = cv2.blur(img, (5, 5))
        seg = self._normalize_img(seg)
        seg = self._roi_net.predict(seg)
        seg = cv2.medianBlur(seg, 5)
        ratio = np.sum(seg) / np.sum(np.ones_like(seg))
        if ratio > 0.5:
            seg = 1.0 - seg
        labels = cv2.dilate(seg, np.ones(3, 3))
        labels = measure.label(labels)
        props = measure.regionprops(labels)

        def get_slices(image: np.ndarray, bbox: tuple) -> np.ndarray:
            assert len(image.shape) == 3
            return image[bbox[0]:bbox[2] - 1, bbox[1]:bbox[3] - 1, :]

        slices = [get_slices(img, prop.bbox) for prop in props]
        predicts = [self._cls_net.predict(slice1) for slice1 in slices]
        result = []
        for pred, prop, slice2 in predicts, props, slices:
            if round(pred):
                result.append((prop, slice2))
        return result

    @staticmethod
    def _normalize_img(img: np.ndarray) -> np.ndarray:
        if img.ndim is not 2:
            img = img.reshape(img.shape[:2])

        adjusted_width = img.shape[0] - img.shape[0] % 32
        adjusted_height = img.shape[1] - img.shape[1] % 32
        img = cv2.resize(img, (adjusted_width, adjusted_height))
        return img.reshape((1,) + img.shape + (1,))


if __name__ == '__main__':
    roi_path = 'E:/Data/ShipDetection/FCN/UNet_768_epoch20.h5'
    cls_path = 'E:/Data/ShipDetection/CNN/model.h5'
    api = SyntheticPipeline(roi_path, cls_path)
    result = api.end2end_predict(cv2.imread('E:/Data/ShipDetection/FCN/large/3.tif', 0))
    print(result)
