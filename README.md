# Construction Defect Detection — Concrete Crack Classification

A PyTorch CNN that classifies concrete surfaces (**Decks, Walls, Pavements**) as **Cracked** or **Non-cracked**, built on the Structural Defects Network (SDNET2018) dataset.

## Overview

Cracks in concrete infrastructure are a key indicator of structural degradation. This project builds an end-to-end image classification pipeline that:

- Loads and consolidates the SDNET2018 dataset across three surface types (Decks, Walls, Pavements)
- Performs EDA on class distribution and surface-type balance
- Trains a custom CNN (`CrackCNN`) to classify images as cracked or non-cracked
- Evaluates performance with accuracy, classification report, and confusion matrix
- Runs inference on a single external image

## Dataset

**Source:** [Structural Defects Network — Concrete Crack Images](https://www.kaggle.com/datasets/aniruddhsharma/structural-defects-network-concrete-crack-images) (Kaggle, via `kagglehub`)

| Property | Value |
|---|---|
| Total images | 56,092 |
| Non-cracked | 47,608 |
| Cracked | 8,484 |
| Surface types | Pavements (24,334), Walls (18,138), Decks (13,620) |
| Image size | 256 × 256 (resized to 224 × 224 for training) |

The dataset is notably **imbalanced** (~85% non-cracked vs. ~15% cracked), which is reflected in the model's per-class metrics below.

## Model Architecture — `CrackCNN`

A lightweight custom CNN built with three convolutional blocks followed by a fully connected classifier:

```
Conv2d(3 → 32, k=3) → ReLU → MaxPool2d(2)
Conv2d(32 → 64, k=3) → ReLU → MaxPool2d(2)
Conv2d(64 → 128, k=3) → ReLU → MaxPool2d(2)
Flatten
Linear(128×28×28 → 512) → ReLU → Dropout(0.5)
Linear(512 → 2)
```

**Training setup:**
- Loss: `CrossEntropyLoss`
- Optimizer: `Adam` (lr = 0.001)
- Epochs: 10
- Batch size: 32
- Data split: 70% train / 15% validation / 15% test (stratified)
- Augmentation (train only): random horizontal flip, random rotation (±10°)
- Normalization: ImageNet mean/std

## Results

| Metric | Score |
|---|---|
| Test Accuracy | **89.9%** |
| Precision (Cracked) | 0.79 |
| Recall (Cracked) | 0.45 |
| F1 (Cracked) | 0.57 |
| Precision (Non-cracked) | 0.91 |
| Recall (Non-cracked) | 0.98 |
| F1 (Non-cracked) | 0.94 |

**Confusion Matrix:**

|            | Pred: Cracked | Pred: Non-cracked |
|------------|:---:|:---:|
| **Actual: Cracked**     | 573 | 700 |
| **Actual: Non-cracked** | 149 | 6992 |

> The model achieves strong overall accuracy but shows lower recall on the minority (Cracked) class due to dataset imbalance — a good candidate for future improvement via class weighting, focal loss, or oversampling.

Training/validation loss and accuracy curves are plotted at the end of the notebook.

## Tech Stack

- **PyTorch** / **torchvision** — model, data loading, transforms
- **scikit-learn** — label encoding, train/val/test split, evaluation metrics
- **pandas** / **NumPy** — data wrangling
- **Matplotlib** / **PIL** — visualization and image handling
- **kagglehub** — dataset download

## Project Structure

```
├── Constructiondefect.ipynb   # Main notebook: EDA → preprocessing → training → evaluation → inference
└── README.md
```

## Getting Started

1. Install dependencies:
   ```bash
   pip install torch torchvision scikit-learn pandas numpy matplotlib pillow kagglehub
   ```
2. Run the notebook top to bottom. The dataset is downloaded automatically via `kagglehub` on first run.
3. GPU (CUDA) is recommended but the notebook falls back to CPU automatically.

## Future Improvements

- Address class imbalance (weighted loss, focal loss, or oversampling of the Cracked class)
- Compare against transfer learning baselines (ResNet, EfficientNet)
- Per-surface-type performance breakdown (Decks vs. Walls vs. Pavements)
- Grad-CAM visualization to localize predicted crack regions

## License

This project uses the SDNET2018 dataset, which is publicly available on Kaggle for research and educational use.

## Read More on : https://majestic-trifle-f5f7e9.netlify.app

