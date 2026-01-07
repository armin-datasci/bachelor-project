import tensorflow as tf
from tensorflow.keras import backend as K

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    """
    Dice coefficient for binary segmentation.
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_loss(y_true, y_pred):
    """
    Dice loss: 1 - Dice coefficient
    """
    return 1 - dice_coefficient(y_true, y_pred)


def bce_dice_loss(y_true, y_pred, alpha=0.5):
    """
    Combined Binary Crossentropy + Dice loss.

    alpha = weight for BCE term
    (1 - alpha) = weight for Dice loss
    """
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)

    return alpha * bce + (1.0 - alpha) * dice
