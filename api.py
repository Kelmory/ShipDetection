from keras import Model
from keras.models import load_model
import os
import cv2
import math
import time
import numpy as np
import pandas as pd
import skimage.measure as measure
import matplotlib.pyplot as plt
from ShipDetection.data_io import PathHelper


class SyntheticPipeline:
    _roi_net: Model
    _cls_net: Model

    def __init__(self, roi_path: str, cls_path: str, pad=30):
        if not PathHelper(roi_path).is_file_valid():
            raise ValueError
        if not PathHelper(cls_path).is_file_valid():
            raise ValueError

        if isinstance(pad, int):
            self.pad = pad
        else:
            self.pad = 30

        self._roi_net = load_model(roi_path, compile=False)
        self._cls_net = load_model(cls_path, compile=False)

        self._roi_net.summary()
        self._cls_net.summary()

    def end2end_predict(self, img: np.ndarray, two_stage=True, mode='mask') -> ([tuple], np.ndarray):
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
            post = img

            post[-40:, :] = 0
            post[:, -40:] = 0

            ratio = np.sum(post) / np.sum(np.ones_like(post))
            if ratio > 0.5:
                post = 1.0 - post
            post = cv2.medianBlur(post, 5)
            post = cv2.dilate(post, np.ones((3, 3)), iterations=5)

            x_off, y_off = 0, 0
            if all((x_off, y_off)):
                new = np.zeros_like(post)
                new[:-x_off, :-y_off] = post[x_off:, y_off:]
                post = new
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

            if (right - left) > 300 or (bottom - top) > 300:
                return None
            elif (bbox[2] - bbox[0]) < (self.pad / 4) or (bbox[3] - bbox[1]) < (self.pad / 4):
                return None
            else:
                return image[left: right, top: bottom]

        def predict_slices(batch: list, threshold: float = 0.9) -> list:
            predicts = []
            for bbox, patch in batch:
                if 0 in patch.shape:
                    continue
                slc = patch.reshape((1, ) + patch.shape + (1, ))
                pred = self._cls_net.predict(slc).sum()
                if pred > threshold:
                    predicts.append((bbox, patch, pred))
            return predicts

        def embedding_slices(predicts: list, img: np.ndarray, mode: str) -> np.ndarray:
            assert img.ndim == 2
            assert mode in ['bbox', 'mask']

            if mode is 'mask':
                new = np.zeros_like(img, dtype=np.uint8)
            else:
                new = img.copy()

            for unit in predicts:
                bbox, patch, conf = unit
                patch = np.round(patch).astype(np.uint8)

                left = bbox[0] - self.pad if (bbox[0] - self.pad) >= 0 else 0
                right = bbox[2] + self.pad if (bbox[2] + self.pad) <= image.shape[0] else bbox[2]
                top = bbox[1] - self.pad if (bbox[1] - self.pad) >= 0 else 0
                bottom = bbox[3] + self.pad if (bbox[3] + self.pad) <= image.shape[1] else bbox[3]

                if mode is 'mask':
                    new[left: right, top: bottom] = patch
                else:
                    cv2.rectangle(new, (top, left), (bottom, right), color=255, thickness=2)
                    cv2.putText(new, 'p: %.2f' % unit[2], (top, left), cv2.FONT_HERSHEY_PLAIN, 1.5, 255, 1)
            if mode is 'mask':
                new = np.clip(img + new * 255 * 0.4, 0, 255)
                new = new.astype(np.uint8)
            return new

        to_predict = pre_process(img)
        roi_img = roi_predict(to_predict)
        roi_img = roi_process(roi_img)
        if two_stage:
            batch = region_props(roi_img)
            slices = predict_slices(batch)
            new = embedding_slices(slices, img, mode=mode)
            return new
        else:
            if mode is 'mask':
                roi_img = np.clip(img + roi_img * 255 * 0.4, 0, 255)
                return roi_img
            else:
                labels = measure.label(roi_img)
                props = measure.regionprops(labels)
                for prop in props:
                    left, top, right, bottom = prop.bbox
                    cv2.rectangle(img, (top, left), (bottom, right), color=255, thickness=2)
                return img


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
        assert batch.ndim is 4

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
    roi_path = 'E:/Data/ShipDetection/FCN/model.h5'
    cls_path = 'E:/Data/ShipDetection/CNN/model_origin.h5'
    api = SyntheticPipeline(roi_path, cls_path, pad=25)

    dir_path = 'E:/Data/ShipDetection/FCN/large/'
    imgs = os.listdir(dir_path)

    tick_list = []

    for i in imgs[20:]:
        print('processing: %s' % i)

        image = cv2.imread(dir_path + i, 0)
        if image is None:
            continue

        tick = time.time()
        result = api.end2end_predict(image, two_stage=False, mode='mask')
        tick2 = time.time()
        tick_list.append((tick2 - tick) * 1000)

        show_img = True
        if show_img:
            plt.imshow(result)
            plt.show()
            plt.cla()

        save = False
        if save:
            img_save = result.astype(dtype=np.uint8)
            cv2.imwrite('E:/Data/ShipDetection/result/img_%s_pad%d.jpg' % (i[:-4], api.pad), img_save)

    pd.DataFrame(tick_list, columns=['time']).to_csv('E:/Data/ShipDetection/result/time.csv', index=False)
