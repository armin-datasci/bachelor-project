import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

def extract_roi(image, mask, margin=10):
    """
    Extract ROI around the lesion using the mask.
    """
    coords = np.where(mask > 0)
    if len(coords[0]) == 0:
        return image, mask  # fallback (rare)

    y_min, y_max = coords[0].min(), coords[0].max()
    x_min, x_max = coords[1].min(), coords[1].max()

    y_min = max(0, y_min - margin)
    x_min = max(0, x_min - margin)
    y_max = min(mask.shape[0], y_max + margin)
    x_max = min(mask.shape[1], x_max + margin)

    return image[y_min:y_max, x_min:x_max], mask[y_min:y_max, x_min:x_max]


def preprocess_dataset(dataset_path, img_height, img_width, img_channel=1):
    """
    ROI-based preprocessing for BUSI dataset.
    """
    X_list, y_list = [], []

    for cls in ["benign", "malignant"]:
        cls_path = os.path.join(dataset_path, cls)
        if not os.path.exists(cls_path):
            continue

        img_files = [
            f for f in os.listdir(cls_path)
            if f.endswith(".png") and "_mask" not in f
        ]

        for img_file in img_files:
            img_path = os.path.join(cls_path, img_file)
            mask_path = os.path.join(
                cls_path, img_file.replace(".png", "_mask.png")
            )

            if not os.path.exists(mask_path):
                continue

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            if img is None or mask is None:
                continue

            # binarize mask
            mask = (mask > 127).astype(np.uint8)

            # extract ROI
            img_roi, mask_roi = extract_roi(img, mask)

            # resize ROI to higher resolution
            img_roi = cv2.resize(img_roi, (img_width, img_height), interpolation=cv2.INTER_LINEAR)
            mask_roi = cv2.resize(mask_roi, (img_width, img_height), interpolation=cv2.INTER_NEAREST)

            img_roi = img_roi.astype(np.float32) / 255.0
            img_roi = np.expand_dims(img_roi, -1)

            mask_roi = np.expand_dims(mask_roi.astype(np.float32), -1)

            X_list.append(img_roi)
            y_list.append(mask_roi)

    X = np.array(X_list)
    y = np.array(y_list)

    print(f"ROI-preprocessed dataset: {X.shape}")
    return X, y


def plot_random_sample(X, y):
    idx = np.random.randint(len(X))
    img = X[idx, :, :, 0]
    mask = y[idx, :, :, 0]

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap="gray")
    plt.title("ROI Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap="gray")
    plt.title("ROI Mask")
    plt.axis("off")
    plt.show()
