# 02-improved-busi-unet
Improved version of the U-Net convolutional neural network for image segmentation on the BUSI dataset.

# Improved U-Net Baseline – BUSI Dataset

## Overview
This experiment presents an improved version of the baseline U-Net model for breast ultrasound image segmentation on the BUSI dataset.  
The goal of this version is to enhance segmentation performance while **keeping the model capacity comparable to the baseline**, ensuring a fair and controlled comparison.

Improvements focus on supervision quality, spatial resolution, and optimization strategy rather than increasing architectural complexity.

---

## Model Architecture
- Architecture: U-Net
- Encoder filters: [24, 48, 96, 192]
- Bottleneck filters: 384
- Decoder: symmetric with skip connections
- Dropout: 0.1
- Trainable parameters: ~4.3 million

The number of parameters is intentionally kept close to the baseline to isolate the effect of training and loss-level improvements.

---

## Input Configuration
- Input image size: 256 × 256
- Channels: 1 (grayscale ultrasound images)
- Normalization: min–max scaling to [0, 1]

---

## Training Configuration
- Optimizer: Adam
- Initial learning rate: 1e-4
- Batch size: same as baseline
- Epochs: 50
- Learning rate scheduler: ReduceLROnPlateau
  - Monitor: validation loss
  - Patience: 10
  - Factor: default
- Early stopping: optional (if enabled, monitors validation loss)

---

## Loss Function
A combined loss function is used to improve supervision on class imbalance and boundary accuracy:

- Binary Cross-Entropy (BCE)
- Soft Dice Loss

Final loss:
Loss = BCE + Dice Loss


---

## Evaluation Metrics
- Dice coefficient
- Binary accuracy
- Validation loss

---

## Motivation for Improvements
Compared to the baseline model, this version introduces:
- Higher input resolution to preserve spatial detail
- Dice-based supervision to better handle class imbalance
- Learning rate scheduling to stabilize convergence

All improvements are designed to enhance performance **without increasing model capacity**, allowing a fair comparison with the baseline.

---

## Notes
This experiment is intended as a methodological improvement over the baseline and serves as a controlled step toward more advanced segmentation models.
