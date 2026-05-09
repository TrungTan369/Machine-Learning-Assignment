<div align="center">
  <img src="https://oisp.hcmut.edu.vn/en/wp-content/uploads/2017/10/HCMUT-BachKhoa-Logo-480x487.png" alt="HCMUT Logo" width="130">

# Course Assignment — Machine Learning (CO3117)

**Assignment 3: Machine Learning with Image Data — INRIA Person**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ngtan369/Hybrid-Image-Classification/blob/main/notebooks/ex3_imageData.ipynb)

</div>

## Course information

| | |
|---|---|
| **Course** | Machine Learning — CO3117 |
| **Semester** | Semester I, academic year 2025–2026 |
| **Faculty** | Faculty of Computer Science and Engineering |
| **University** | Ho Chi Minh City University of Technology, VNU-HCM |
| **Instructor** | Dr. Le Thanh Sach |


## Objectives

This repository contains the end-to-end image classification pipeline for **Assignment 3** of the course, evaluated on the **INRIA Person** dataset:

```
EDA  →  Build dataset  →  Deep feature extraction  →  Classifier comparison  →  Best-model analysis  →  (Bonus) End-to-end VGG16
```

The pipeline meets the requirements of the assignment:

- Basic EDA: image sizes, colour channels, label distribution, sample visualization with bounding boxes.
- Deep feature extraction from pretrained models (ResNet50 / VGG16 / EfficientNetB0), saved as `.npy` files.
- Comparison of multiple traditional machine-learning classifiers (Logistic Regression / Linear SVM / Random Forest).
- Bonus extension (extra credit): end-to-end deep learning pipeline with VGG16 (transfer learning + fine-tuning).
- The whole pipeline is configured through a single `CONFIG` dict inside the notebook.

## How to run

### Google Colab (recommended)

1. Click the **Open in Colab** badge at the top of this README, or open `notebooks/ex3_imageData.ipynb` directly from the repo.
2. If GPU is needed: `Runtime → Change runtime type → GPU` (the free-tier T4 is enough).
3. `Runtime → Run all`.

The first cell automatically:

- Clones the repo into `/content/Hybrid-Image-Classification`.
- Adds `modules/` to `sys.path`.
- Installs the missing libraries (`kagglehub`, `scikit-learn`, `Pillow`, `seaborn`).

The data-download cell uses `kagglehub.dataset_download("jcoral02/inriaperson")`. On a fresh Colab (without Kaggle credentials), the helper already includes a workaround that bypasses the **Colab cache resolver** — sidestepping the case where kagglehub hangs. If Kaggle returns 401, add `KAGGLE_USERNAME` / `KAGGLE_KEY` to Colab Secrets (the key icon in the left sidebar) and re-run the cell.

**No Google Drive mount required.**

### Run locally (Python ≥ 3.10)

```bash
git clone https://github.com/ngtan369/Hybrid-Image-Classification.git
cd Hybrid-Image-Classification
pip install -r requirement.txt
jupyter lab notebooks/ex3_imageData.ipynb
```

Main library requirements (full list in `requirement.txt`):

- `numpy`, `pandas`, `Pillow`, `matplotlib`, `seaborn`
- `scikit-learn` — traditional classifiers
- `tensorflow` — pretrained CNN backbones + end-to-end training
- `kagglehub` — dataset download

## Pipeline configuration

All parameters live in the `CONFIG` dict in cell 4 of the notebook:

```python
CONFIG = {
    "image_size":        (224, 224),
    "test_size":         0.2,
    "random_state":      42,
    "voc_neg_per_image": 5,                                        # negatives per image
    "voc_target_label":  "person",
    "feature_models":    ["resnet50", "vgg16", "efficientnetb0"],
    "classifiers":       ["logreg", "svm_linear", "random_forest"],
    "dl_epochs_head":    3,                                        # transfer learning
    "dl_epochs_ft":      2,                                        # fine-tuning
    "dl_unfreeze":       4,                                        # last N VGG16 layers
}
```

## Project structure

```text
Hybrid-Image-Classification/
├── notebooks/
│   └── ex3_imageData.ipynb     # Front-end Colab notebook (end-to-end run)
├── modules/
│   ├── ml_utils.py             # Dataset discovery, EDA, classifier comparison,
│   │                           # VOC bbox helpers, feature I/O (TensorFlow-free)
│   └── dl_utils.py             # Pretrained feature extraction, transfer-learning
│                               # helpers (TF lazy-imported, Keras 3 compatible)
├── features/                   # Feature vectors .npy (generated when running the notebook)
├── reports/
│   └── report.pdf              # PDF report (rendered from report.md)
├── mlAssignments_v1.1.pdf      # Assignment specification
├── requirement.txt             # Pip dependencies
└── README.md                   # This file
```

## Results summary

On the INRIA Person dataset (902 source images → 2973 samples after bbox cropping and negative sampling):

| Configuration | Accuracy | F1-macro |
|---|---|---|
| **ResNet50 + Linear SVM** *(best)* | ~0.94 | ~0.93 |
| EfficientNetB0 + Linear SVM | ~0.93 | ~0.92 |
| VGG16 + Linear SVM | ~0.89 | ~0.88 |
| VGG16 end-to-end (3 + 2 epoch) | ~0.87 | ~0.87 |
