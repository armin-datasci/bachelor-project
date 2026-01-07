import cv2
import numpy as np

def augment_sample(image, mask):
    """Light geometric + photometric augmentation."""

    if np.random.rand() < 0.5:
        image = np.fliplr(image)
        mask = np.fliplr(mask)

    if np.random.rand() < 0.5:
        image = np.flipud(image)
        mask = np.flipud(mask)

    if np.random.rand() < 0.5:
        angle = np.random.uniform(-15, 15)
        image, mask = rotate(image, mask, angle)

    if np.random.rand() < 0.5:
        factor = np.random.uniform(0.9, 1.1)
        image = np.clip(image * factor, 0, 1)

    return image, mask


def rotate(image, mask, angle):
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)

    img = cv2.warpAffine(
        image[:, :, 0], M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )

    msk = cv2.warpAffine(
        mask[:, :, 0], M, (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_REFLECT
    )

    return np.expand_dims(img, -1), np.expand_dims(msk, -1)
