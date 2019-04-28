import tensorflow as tf
import keras.backend as K
import numpy as np
from keras.losses import binary_crossentropy


# Losses
def dice_p_bce(in_gt, in_pred):
    return 1e-3 * binary_crossentropy(in_gt, in_pred) - dice_coef(in_gt, in_pred)


def focal_loss(gamma=2., alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(K.equal(y_true, 1), y_pred, K.ones_like(y_pred))
        pt_0 = tf.where(K.equal(y_true, 0), y_pred, K.zeros_like(y_pred))
        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) \
               - K.sum((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
    return focal_loss_fixed


def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true), -1) + K.sum(K.square(y_pred), -1) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


# Metrics
def true_positive_rate(y_true, y_pred):
    if np.max(y_true) == 0.0:
        return true_negative_rate(1 - y_true, 1 - y_pred)
    return (K.sum(K.flatten(y_pred) * K.flatten(y_true)) + 1e-3) / (K.sum(K.flatten(y_true)) + 1e-3)


def true_negative_rate(y_true, y_pred):
    return (K.sum(K.flatten(1 - y_pred) * K.flatten(1 - y_true))) / (K.sum(K.flatten(1 - y_true)) + 1e-3)


def false_negative_rate(y_true, y_pred):
    return 1 - true_positive_rate(y_true, y_pred)


def IoU(y_true, y_pred, eps=1e-6):
    if np.max(y_true) == 0.0:
        return IoU(1-y_true, 1-y_pred)
    intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true) + K.sum(y_pred) - intersection
    return -K.mean((intersection + eps) / (union + eps))
