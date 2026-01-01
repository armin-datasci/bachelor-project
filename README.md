# Breast Ultrasound Image Segmentation Using U-Net Architectures

## Abstract

This repository presents the refactored implementation of my bachelor’s thesis project, which focuses on the semantic segmentation of breast lesions in ultrasound images using deep learning techniques. The study is conducted on the publicly available Breast Ultrasound Images (BUSI) dataset and employs the U-Net architecture as a baseline model, followed by an improved variant with modified architectural and training choices. The objective of this project is to investigate the effectiveness of convolutional neural networks for medical image segmentation within the scope of an undergraduate thesis, emphasizing methodological clarity, reproducibility, and controlled performance evaluation.

---

## 1. Introduction

Semantic segmentation plays a central role in medical image analysis, particularly for tasks requiring precise localization of anatomical structures or pathological regions. In breast ultrasound imaging, accurate delineation of lesion boundaries can support diagnostic workflows by providing quantitative and visual information to clinicians. However, ultrasound images are characterized by low contrast, speckle noise, and high variability in lesion appearance, which makes automated segmentation a challenging problem.

U-Net is a widely adopted convolutional neural network architecture designed for biomedical image segmentation. Its encoder–decoder structure with skip connections enables the extraction of contextual information while preserving fine-grained spatial details. In this project, U-Net is used as a baseline model, and a modified version is developed to explore the impact of architectural and training adjustments on segmentation performance.

---

## 2. Dataset

### BUSI Dataset

The Breast Ultrasound Images (BUSI) dataset is a publicly available dataset introduced by Al-Dhabyani et al. (2020). It contains grayscale ultrasound images of breast tissue along with expert-annotated ground truth masks.

* Total images: 780
* Classes: normal, benign, malignant
* Ground truth: pixel-wise binary masks for lesion regions

In this project, only benign and malignant images are used, as the normal class does not contain lesion masks and is therefore unsuitable for supervised segmentation.

### Data Preparation

* Images and masks are resized to a fixed spatial resolution (128×128 for the baseline model and 256×256 for the improved model).
* Pixel intensities are normalized to the range [0, 1].
* Masks are binarized to represent lesion versus background.

The dataset is split into training and validation subsets using a fixed random seed to ensure reproducibility.

---

## 3. Methodology

### 3.1 Baseline Model: U-Net

The baseline model follows the original U-Net design:

* Encoder–decoder architecture with symmetric downsampling and upsampling paths
* Convolutional blocks consisting of convolution, activation, and dropout layers
* Input resolution: 128×128
* Dropout rate: 0.1
* Loss function: Binary Cross-Entropy

This configuration serves as a reference point for evaluating subsequent improvements.

### 3.2 Model Complexity and Parameter Count

To provide a transparent comparison between the baseline and improved architectures, the total number of trainable parameters for each model is reported below.

| Model Variant  | Input Resolution | Trainable Parameters |
| -------------- | ---------------- | -------------------- |
| Baseline U-Net | 128×128          | ~4.3 million         |
| Improved U-Net | 256×256          | ~ million          |

The increase in parameter count is primarily due to the higher input resolution and corresponding feature map sizes throughout the encoder–decoder pathway. This comparison clarifies the trade-off between model capacity and computational complexity. The reported parameter counts were obtained directly using the Keras `model.summary()` method.

---

### 3.3 Improved Model

The improved U-Net variant introduces several modifications aimed at enhancing segmentation performance:

* Increased input resolution to 256×256 to preserve spatial detail
* Reduced dropout rate (0.05) to limit underfitting
* Combined loss function: Binary Cross-Entropy + Dice Loss
* Dice coefficient used as an additional validation metric

These changes are motivated by the characteristics of medical image segmentation, where class imbalance and boundary precision are critical considerations.

---

## 4. Training Procedure

Both models are trained under controlled experimental conditions:

* Optimizer: Adam
* Initial learning rate: 1e-4
* Batch size: defined according to GPU memory constraints
* Number of epochs: fixed upper bound with early stopping
* Callbacks: early stopping and model checkpointing based on validation loss

All experiments are conducted using the same train–validation split to ensure a fair comparison between models.

---

## 5. Results and Evaluation

### Evaluation Metrics and Loss Interpretation

The models are trained and evaluated using loss functions and metrics that are standard for binary semantic segmentation tasks in medical imaging. For the baseline model, Binary Cross-Entropy (BCE) loss is employed, while the improved model uses a composite loss defined as the sum of Binary Cross-Entropy and Dice Loss. This combined formulation balances pixel-wise classification accuracy with region-level overlap, which is particularly important in the presence of class imbalance between lesion and background pixels.

During evaluation, the primary reported metrics are Mean Intersection over Union (Mean IoU) and the Dice coefficient. Mean IoU measures the overlap between predicted and ground truth masks relative to their union, while the Dice coefficient emphasizes agreement on the segmented lesion region. These metrics provide complementary perspectives on segmentation quality and are more informative than accuracy alone for pixel-wise prediction tasks.

It is important to note that the observed loss values are relatively low and the accuracy values relatively high due to the strong class imbalance inherent in ultrasound segmentation, where background pixels dominate the images. In such settings, accuracy can be inflated by correct background classification and should therefore not be interpreted as a standalone indicator of segmentation performance. For this reason, overlap-based metrics such as Dice score and IoU are emphasized throughout this project.

The performance of the baseline and improved models is evaluated using standard segmentation metrics.

| Metric     | Baseline U-Net | Improved U-Net |
| ---------- | -------------- | -------------- |
| Mean IoU   | ~0.72          | ~0.47          |
| Dice Score | –              | ~0.70          |

The improved model demonstrates a modest but consistent improvement over the baseline, indicating that higher input resolution and an appropriate loss function contribute positively to segmentation quality. All reported metrics are computed on the validation set and are intended for comparative evaluation only.

Qualitative results, including predicted masks overlaid on input images, are provided in the repository to visually assess model behavior.

---

## 6. Usage and Reproducibility

### Requirements

* Python 3.9
* TensorFlow / Keras (2.x)
* NumPy
* Matplotlib

### Installation

```bash
git clone https://github.com/armin-datasci/bachelor-project.git
cd bachelor-project
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Training

```bash
python train.py
```

Ensure that the dataset directory follows the structure described in the dataset section before running the training scripts.

---

## 7. Limitations

This project is conducted within the scope of an undergraduate thesis and is subject to several limitations:

* Limited dataset size
* No cross-dataset generalization analysis
* No clinical validation or deployment considerations

The results should therefore be interpreted as exploratory and educational rather than clinically actionable.

---

## 8. Conclusion

This project demonstrates the application of U-Net-based architectures to breast ultrasound image segmentation and highlights the impact of architectural and training refinements. Through systematic experimentation, the study provides practical insight into deep learning workflows for medical image analysis at the bachelor’s level.

---

## 9. References

* Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation.
* Al-Dhabyani, W., Gomaa, M., Khaled, H., & Fahmy, A. (2020). Dataset of breast ultrasound images.
