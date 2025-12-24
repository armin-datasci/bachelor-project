# Bachelor's Project: Breast Ultrasound Segmentation with UNet
## Overview
This project focuses on semantic segmentation of breast ultrasound images using UNet-based architectures. The goal is to explore the baseline model, improve its performance, and provide a detailed comparison between versions. The dataset used is the BUSI dataset containing benign, malignant, and normal breast images (normal images are not used in segmentation).

This repository contains two versions of the project:
  - Baseline UNet – Original UNet implementation.
  - Improved UNet – Enhanced version with adjusted hyperparameters, increased input resolution, modified dropout, and custom loss functions (BCE + Dice Loss and Soft Dice Coefficient).


## Dataset
  - BUSI Dataset
      - Subfolders: benign/, malignant/, normal/ (normal images are not used)
      - Image naming: e.g., benign(1).png and corresponding mask benign(1)_mask.png
  
  - Preprocessing
      - Images resized to 256×256 for improved UNet.
      - Grayscale conversion.
      - Optional data augmentation (rotation, flips).
   

## Architecture
  - Baseline UNet
      - Input resolution: 128×128
      - Standard UNet encoder-decoder with skip connections
      - Convolution blocks with ReLU activations
      - Dropout: 0.2


## Loss Functions and Metrics
  - Binary Cross-Entropy + Dice Loss (bce_dice_loss) – Combines pixel-wise loss and region-based overlap.
  - Soft Dice Coefficient (soft_dice_coef) – Smooth, differentiable version of Dice, used for monitoring.
  - Mean IoU – Measures overlap between predicted mask and ground truth.


## Training Pipeline
  - Optimizer: Adam
  - Learning rate: 2e-4 → 5e-5 (adaptive with ReduceLROnPlateau)
  - Epochs: 50 (early stopping applied to reduce overfitting)
  - Batch size: 8
  - Validation split: 0.2 (higher validation samples for robust evaluation)
  - Callbacks:
    - EarlyStopping – stops training when validation loss stops improving.
    - ReduceLROnPlateau – decreases LR on plateau to stabilize training.
    - DisplayPrediction – visualizes predictions after each epoch.


## Comparison: Baseline vs Improved

| Metric           | Baseline        | Improved                       | Notes                                        |
| ---------------- | --------------- | ------------------------------ | -------------------------------------------- |
| Input resolution | 128×128         | 256×256                        | Higher resolution improves spatial details   |
| Dropout          | 0.2             | 0.05                           | Slight reduction to retain feature learning  |
| Parameters       | ~3M             | ~6M                            | Increased capacity to model complex features |
| Loss             | BCE + Dice      | BCE + Dice + Soft Dice         | Improved monitoring with soft Dice           |
| Mean IoU (val)   | ~45%            | ~47%                           | Slight improvement, more stable curves       |
| Soft Dice (val)  | N/A             | ~0.7                           | New metric for region-based overlap          |
| Training curves  | Loss & Accuracy | Loss, Accuracy, IoU, Soft Dice | Improved visualization                       |


## Notes on improvements:
  - Increased input size for better segmentation detail.
  - Reduced dropout to preserve learning capacity.
  - Added soft dice coefficient for better convergence monitoring.
  - Adjusted learning rate scheduling for stable training curves.
  - Results are smoother and more meaningful than baseline.


## Figures and Visualization
  - All training metrics are visualized using plot_history() and saved via save_history_figures().
  - Inline display of predictions with DisplayPrediction callback.
  - Stored in /figures for easy reference and comparison.


## Usage
  - Clone repository:
    -  git clone https://github.com/armin-datasci/bachelors-project.git
  - Navigate to the version you want:
    -  cd 02-improved-busi-unet
  - Train the model:
    -  from src.train_pipeline import train_pipeline
    -  from src.model import build_unet
    -  from src.dataset_loader import load_busi_dataset
    -  X, y = load_busi_dataset(dataset_path, img_size=(256,256))
    -  X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    -  model = build_unet(input_shape=(256,256,1), dropout_rate=0.05)
    -  model, history = train_pipeline(X_train, y_train, X_val, y_val, model=model)
  - Plot and save figures:
    - from src.utils import plot_history, save_history_figures
    - plot_history(history)
    - save_history_figures(history, fig_folder="figures")


## Key Takeaways
  - Clear baseline → improved workflow demonstrates iterative improvement.
  - Documented architecture, loss functions, hyperparameters, and training pipeline.
  - Figures show stable, meaningful curves for loss, accuracy, IoU, and soft Dice.
  - The project demonstrates research rigor, experimental reproducibility, and deep learning proficiency suitable for MSc Data Science applications.
