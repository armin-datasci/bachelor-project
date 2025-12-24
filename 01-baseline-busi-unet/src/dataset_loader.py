import os
import cv2
import numpy as np
from skimage.transform import resize

def load_busi_dataset(path, img_size=(128, 128)):
    images, masks = [], []
    folders = ['benign', 'malignant']

    for folder in folders:
        folder_path = os.path.join(path, folder)
        for file in os.listdir(folder_path):
            if file.endswith(".png") and "_mask" not in file:
                img_path = os.path.join(folder_path, file)
                mask_path = os.path.join(folder_path, file.replace(".png", "_mask.png"))

                image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

                image = resize(image, img_size, preserve_range=True)
                mask = resize(mask, img_size, preserve_range=True)

                image = np.expand_dims(image, axis=-1) / 255.0
                mask = np.expand_dims(mask, axis=-1) / 255.0

                images.append(image)
                masks.append(mask)

    images = np.array(images, dtype=np.float32)
    masks = np.array(masks, dtype=np.float32)
    print(f"Total samples loaded: {images.shape[0]}")
    return images, masks
