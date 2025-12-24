import os
import cv2
import numpy as np

def load_busi_dataset(path, img_size=(256,256), augment=False):
    images = []
    masks = []

    for category in ["benign", "malignant"]:
        cat_path = os.path.join(path, category)
        if not os.path.exists(cat_path):
            continue

        for filename in os.listdir(cat_path):
            if filename.endswith(".png") and "_mask" not in filename:
                # Corresponding mask
                mask_name = filename.replace(".png", "_mask.png")
                img_path = os.path.join(cat_path, filename)
                mask_path = os.path.join(cat_path, mask_name)

                if not os.path.exists(img_path) or not os.path.exists(mask_path):
                    continue

                # Load image and mask as grayscale
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

                # Resize
                img = cv2.resize(img, img_size)
                mask = cv2.resize(mask, img_size)

                # Normalize
                img = img.astype(np.float32) / 255.0
                mask = mask.astype(np.float32) / 255.0

                # Expand dims for channel
                img = np.expand_dims(img, axis=-1)
                mask = np.expand_dims(mask, axis=-1)

                images.append(img)
                masks.append(mask)

    if len(images) == 0:
        print("Warning: Loaded 0 images and masks from BUSI dataset.")
        return np.array([]), np.array([])

    images = np.array(images)
    masks = np.array(masks)

    # data augmentation
    if augment:
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        datagen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True
        )
        images = np.array([datagen.random_transform(img) for img in images])
        masks = np.array([datagen.random_transform(mask) for mask in masks])

    print(f"Loaded {len(images)} images and masks from BUSI dataset.")
    return images, masks
