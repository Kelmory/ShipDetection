from keras import Model
from keras.models import load_model
import cv2
import math
import numpy as np
import skimage.measure as measure
import matplotlib.pyplot as plt
from ShipDetection.data_io import PathHelper


class SyntheticPipeline:
    _roi_net: Model
    _cls_net: Model

    pad = 30

    def __init__(self, roi_path, cls_path):
        if not PathHelper(roi_path).is_file_valid():
            raise ValueError
        if not PathHelper(cls_path).is_file_valid():
            raise ValueError
        self._roi_net = load_model(roi_path, compile=False)
        self._cls_net = load_model(cls_path, compile=False)

    def end2end_predict(self, img: np.ndarray) -> ([tuple], np.ndarray):
        def pre_process(img: np.ndarray) -> np.ndarray:
            pre = cv2.blur(img, (5, 5))
            return pre

        def roi_predict(img: np.ndarray) -> np.ndarray:
            roi = self._normalize_img(img, self._roi_net.input_shape)
            predict = []
            for i in range(roi.shape[0]):
                pre = self._roi_net.predict(roi[i, :, :, :].reshape((1,) + roi.shape[1:]))[0, :, :, :]
                predict.append(pre)
            roi = np.asarray(predict)
            roi = self._flatten_img(roi, img.shape)
            return roi.astype(np.float32)

        def roi_process(img: np.ndarray) -> np.ndarray:
            post = cv2.medianBlur(img, 5)
            ratio = np.sum(post) / np.sum(np.ones_like(post))
            if ratio > 0.5:
                post = 1.0 - post
            return post

        def region_props(img: np.ndarray) -> list:
            labels = measure.label(img)
            props = measure.regionprops(labels)
            slices = []
            for prop in props:
                patch = get_slices(img, prop.bbox)
                if patch is not None:
                    slices.append((prop.bbox, patch))
            return slices

        def get_slices(image: np.ndarray, bbox: tuple) -> [np.ndarray, None]:
            assert image.ndim == 2

            left = bbox[0] - self.pad if (bbox[0] - self.pad) >= 0 else 0
            right = bbox[2] + self.pad if (bbox[2] + self.pad) <= image.shape[0] else bbox[2]

            top = bbox[1] - self.pad if (bbox[1] - self.pad) >= 0 else 0
            bottom = bbox[3] + self.pad if (bbox[3] + self.pad) <= image.shape[1] else bbox[3]

            if (right - left) > 400 or (top - bottom) > 400:
                return None
            else:
                return image[left: right, top: bottom]

        def predict_slices(batch: list, threshold: float = 0.4) -> list:
            predicts = []
            for bbox, patch in batch:
                slc = patch.reshape((1, ) + patch.shape + (1, ))
                pred = self._cls_net.predict(slc).sum()
                if pred > threshold:
                    predicts.append((bbox, patch, pred))
            return predicts

        def embedding_slices(predicts: list, img: np.ndarray) -> np.ndarray:
            assert img.ndim == 2

            new = np.zeros_like(img, dtype=np.float32)
            for unit in predicts:
                bbox = unit[0]
                patch = unit[1]
                new[bbox[0]:bbox[2], bbox[1]:bbox[3]] += patch[self.pad: -self.pad, self.pad: -self.pad]
            return new

        to_predict = pre_process(img)
        roi_img = roi_predict(to_predict)
        roi_img = roi_process(roi_img)
        batch = region_props(roi_img)
        slices = predict_slices(batch)
        new = embedding_slices(slices, img)
        return new

    @staticmethod
    def _normalize_img(img: np.ndarray, shape: (int, int, int, int)) -> np.ndarray:
        if img.ndim is not 2:
            img = img.reshape(img.shape[:2])

        if not any(shape):
            return img
        else:
            shape = shape[1: -1]

        adjusted_width = math.ceil(img.shape[0] / shape[0]) * shape[0]
        adjusted_height = math.ceil(img.shape[1] / shape[1]) * shape[1]

        padding = np.zeros((adjusted_width, adjusted_height))
        padding[:img.shape[0], :img.shape[1]] = img[:, :]

        row = math.ceil(img.shape[1] / shape[1])
        col = math.ceil(img.shape[0] / shape[0])

        batch = np.zeros((row * col, ) + shape + (1,))
        for i in range(row * col):
            r, c = int(i / row), int(i % row)
            batch[i, :, :, 0] = padding[r * shape[0]: (r + 1) * shape[0], c * shape[1]: (c + 1) * shape[1]]

        return batch

    @staticmethod
    def _flatten_img(batch: np.ndarray, img_shape: (int, int)) -> np.ndarray:
        if batch.ndim is not 4:
            raise ValueError

        patch_width, patch_height = batch.shape[1], batch.shape[2]
        col, row = math.ceil(img_shape[0] / patch_height), math.ceil(img_shape[1] / patch_width)

        padding_width = math.ceil(img_shape[0] / patch_width) * patch_width
        padding_height = math.ceil(img_shape[1] / patch_height) * patch_height

        patch = np.zeros((padding_width, padding_height))

        for i in range(batch.shape[0]):
            r, c = int(i / col), int(i % col)
            patch[patch_width * r: patch_width * (r + 1), patch_height * c: patch_height * (c + 1)] = batch[i, :, :, 0]

        img = patch[:img_shape[0], :img_shape[1]]
        return img


if __name__ == '__main__':
    roi_path = 'E:/Data/ShipDetection/FCN/UNet_768_epoch20.h5'
    cls_path = 'E:/Data/ShipDetection/CNN/model.h5'
    api = SyntheticPipeline(roi_path, cls_path)

    image = cv2.imread('E:/Data/ShipDetection/FCN/large/10.tif', 0)
    result = api.end2end_predict(image)
    plt.imshow(image)
    plt.imshow(result, alpha=0.4)
    plt.show()
