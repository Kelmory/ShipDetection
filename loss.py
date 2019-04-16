import tensorflow as tf
import keras.backend as K
import numpy as np
from keras.losses import binary_crossentropy


# Losses
def dice_p_bce(in_gt, in_pred):
    return 1e-3 * binary_crossentropy(in_gt, in_pred) - dice_coef(in_gt, in_pred)


def focal_loss_fixed(y_true, y_pred, alpha=0.25, gamma=2):
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - \
        K.sum((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))


def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true), -1) + K.sum(K.square(y_pred), -1) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


# Metrics
def true_positive_rate(y_true, y_pred):
    return (K.sum(K.flatten(y_pred) * K.flatten(y_true)) + 1e-3) / (K.sum(K.flatten(y_true)) + 1e-3)


def true_negative_rate(y_true, y_pred):
    return (K.sum(K.flatten(1 - y_pred) * K.flatten(1 - y_true)) + 1e-3) / (K.sum(K.flatten(1 - y_true)) + 1e-3)


def false_negative_rate(y_true, y_pred):
    return 1 - true_positive_rate(y_true, y_pred)


def IoU(y_true, y_pred):
    if np.max(y_true) == 0.0:
        return IoU(1 - y_true, 1 - y_pred)
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3]) - intersection
    return -K.mean((intersection + K.epsilon()) / (union + K.epsilon()), axis=0)
