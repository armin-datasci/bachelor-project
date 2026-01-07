import numpy as np
from tensorflow.keras.metrics import MeanIoU

def mean_iou(y_true, y_pred, num_classes=2, threshold=0.5):
    """
    Compute Mean IoU for the offline evaluation.

    Args:
        y_true: ground truth masks, shape (N, H, W, 1)
        y_pred: predicted masks, shape (N, H, W, 1)
        num_classes: number of classes (binary: 2)
        threshold: threshold to binarize predictions

    Returns:
        mean IoU value
    """
    # Binarize predictions
    y_pred_bin = (y_pred >= threshold).astype(np.uint8)
    y_true_bin = y_true.astype(np.uint8)

    # Flatten to compute MeanIoU
    y_true_flat = y_true_bin.flatten()
    y_pred_flat = y_pred_bin.flatten()

    m = MeanIoU(num_classes=num_classes)
    m.update_state(y_true_flat, y_pred_flat)
    return m.result().numpy()
