import numpy as np

def mean_iou(y_true, y_pred, threshold=0.5, eps=1e-6):
    """
    Compute Mean IoU over a dataset.
    """
    y_pred = (y_pred > threshold).astype(np.float32)
    y_true = y_true.astype(np.float32)

    intersection = np.sum(y_true * y_pred, axis=(1, 2, 3))
    union = np.sum(y_true + y_pred, axis=(1, 2, 3)) - intersection

    iou = (intersection + eps) / (union + eps)
    return np.mean(iou)
