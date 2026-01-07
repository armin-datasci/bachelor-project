import os
import cv2
import numpy as np
from skimage.transform import resize


def load_busi_dataset(path, img_size=(192, 192)):
    """
    Load BUSI dataset for the IMPROVED model.

    - Images resized to 192x192
    - Masks resized with nearest-neighbor interpolation
    - Intensity normalized to [0, 1]
    """

    images, masks = [], []
    folders = ["benign", "malignant"]

    for folder in folders:
        folder_path = os.path.join(path, folder)

        for file in os.listdir(folder_path):
            if file.endswith(".png") and "_mask" not in file:
                img_path = os.path.join(folder_path, file)
                mask_path = os.path.join(
                    folder_path, file.replace(".png", "_mask.png")
                )

                # Load grayscale image and mask
                image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

                # Resize image (bilinear) and mask (nearest-neighbor)
                image = resize(
                    image,
                    img_size,
                    preserve_range=True,
                    anti_aliasing=True
                )

                mask = resize(
                    mask,
                    img_size,
                    preserve_range=True,
                    order=0,
                    anti_aliasing=False
                )

                # Normalize and add channel dimension
                image = np.expand_dims(image, axis=-1) / 255.0
                mask = np.expand_dims(mask, axis=-1) / 255.0

                images.append(image)
                masks.append(mask)

    images = np.asarray(images, dtype=np.float32)
    masks = np.asarray(masks, dtype=np.float32)

    print(f"Loaded {images.shape[0]} samples.")
    return images, masks
