# 01-baseline-busi-unet
Baseline version of a U-Net convolutional neural network for image segmentation on the BUSI dataset.

## Baseline U-Net Configuration

### Model Parameters
- Input shape: 128x128x1
- Encoder filters: [16, 32, 64, 128]
- Bottleneck filters: 256
- Decoder filters: [128, 64, 32, 16]
- Dropout rate: 0.2
- Total parameters: ~4.3M
- Output: 1 channel, sigmoid

### Dataset
- BUSI (Breast Ultrasound Images)
- Train/Validation split: 80/20
- Image size: 128x128, grayscale
- Normalization: [0,1]

### Training
- Batch size: 8
- Epochs: 50 (early stopping at 36)
- Learning rate: 1e-4 (effectively 5e-5 after ReduceLROnPlateau)
- Loss: Binary Crossentropy
- Metrics: Accuracy, MeanIoU
- Optimizer: Adam
- Callbacks: EarlyStopping (patience=7), ReduceLROnPlateau

### Performance
| Metric | Train | Validation |
|--------|-------|------------|
| Accuracy | 0.9585 | 0.9241 |
| Loss | 0.1168  | 0.3481 |
| Mean IoU | 0.7809 | 0.7249 |
