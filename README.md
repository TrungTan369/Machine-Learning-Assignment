<div align="center">
  <img src="https://oisp.hcmut.edu.vn/en/wp-content/uploads/2017/10/HCMUT-BachKhoa-Logo-480x487.png" alt="HCMUT Logo" width="150">
  
  # HO CHI MINH CITY UNIVERSITY OF TECHNOLOGY
  ## FACULTY OF COMPUTER SCIENCE & ENGINEERING
  
  ### 📚 Course: Machine Learning (Septem Spring 252)
  ### 👨‍🏫 Mentor: Truong Vinh Lan
</div>

<br>

---

## 🎯 Objectives
This project aims to achieve the following educational goals:
* Understand and apply the traditional machine learning pipeline, including: data preprocessing, feature extraction, model training, and evaluation.
* Practice skills in deploying machine learning models on different types of data: tabular, text, and image.
* Develop the ability to analyze, compare, and evaluate the effectiveness of machine learning models through performance metrics.
* Enhance programming, experimenting, and scientific report organization skills.

---

## 🚀 How to Run the Notebooks

### 1. Requirements & Libraries
The project uses Python 3.x. All required libraries are listed in the `requirement.txt` file. Key libraries include:
* `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`
* `kagglehub` (for downloading the dataset)
* `opencv-python` (for image processing)

**To install the dependencies locally:**
```bash
pip install -r requirement.txt
```

### 2. Execution on Google Colab (Recommended)
Our notebooks are optimized for the Google Colab environment.

1. Open the desired notebook from this repository or upload it to Colab.
2. In the Colab menu, select **Runtime > Change runtime type** and choose **GPU** (especially for the image notebook).
3. Select **Runtime > Run all** to execute the entire pipeline.

**Note on Data:** The notebooks can be configured to automatically download datasets from public sources (e.g., Kaggle) directly into the Colab server's storage using `kagglehub`. No personal Google Drive mounting is required.

---

## 📂 Project Structure

```text
ML_Assignment_HCMUT/
│
├── notebooks/             # Google Colab / Jupyter notebooks
│   └── ex3_imageData.ipynb
│
├── modules/               # Core Python scripts and helper functions
│   └── image_utils.py     # Image preprocessing, feature extraction, etc.
│
├── features/              # Extracted feature files (.npy, .h5, ...)
│   └── bai3.npy
│
├── reports/               # Final project reports
│   └── report.pdf
│
├── requirement.txt        # Project dependencies
└── README.md              # Project documentation
```

---

## 🔗 Project Artifacts

- 📑 Final Report (PDF): `reports/report.pdf`
- 💻 Image Notebook: `notebooks/ex3_imageData.ipynb`
