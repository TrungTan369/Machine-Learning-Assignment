#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
=======================================================================
  PEDESTRIAN DETECTION PIPELINE — INRIAPerson Dataset
=======================================================================

A complete Machine Learning pipeline for pedestrian detection in images.

Pipeline Steps
--------------
  1. Setup & Data Loading
  2. XML Parsing & ROI Extraction
  3. Feature Extraction  (HOG  or  Pre-trained CNN — configurable)
  4. Classifier Training  (LinearSVC + Hard Negative Mining)
  5. Object Detection Inference  (Image Pyramid + Sliding Window)
  6. Post-processing (NMS) & Visualization

Designed for Google Colab. Run cells sequentially.
Dataset: Kaggle — jcoral02/inriaperson
"""

# ╔══════════════════════════════════════════════════════════════════════╗
# ║  BLOCK 1 — Setup & Data Loading                                    ║
# ╚══════════════════════════════════════════════════════════════════════╝

# --- 1.1  Install dependencies & download dataset ---
# (Uncomment the pip/kagglehub lines when running on Colab)

# !pip install kagglehub scikit-image opencv-python-headless -q

import os
import sys
import random
from pathlib import Path

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage import color, feature

from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

print("Core libraries imported ✓")

# --- 1.2  Download dataset via kagglehub ---
import kagglehub

print("Downloading INRIA Person dataset from Kaggle …")
dataset_path = kagglehub.dataset_download("jcoral02/inriaperson")
print(f"Dataset downloaded → {dataset_path}")

# --- 1.3  Clone repo & add modules to path ---
REPO_URL = "https://github.com/ngtan369/Hybrid-Image-Classification"
REPO_DIR = "/content/repo"

if not Path(REPO_DIR).exists():
    os.system(f"git clone {REPO_URL} {REPO_DIR}")

if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from modules.ml_utils import (
    parse_annotations,
    extract_positive_samples,
    extract_negative_samples,
    hog_features,
    hog_feature_single,
    sliding_window,
    image_pyramid,
    non_max_suppression,
    save_features,
    load_features,
)
from modules.dl_utils import cnn_features, cnn_feature_single

print("Custom modules loaded ✓")

# --- 1.4  Discover dataset structure ---

def find_subdirectory(root, target_name):
    """Recursively search for a directory named *target_name* under *root*."""
    for dirpath, dirnames, _ in os.walk(root):
        for d in dirnames:
            if d.lower() == target_name.lower():
                return os.path.join(dirpath, d)
    return None

ANNOTATIONS_DIR = find_subdirectory(dataset_path, "Annotations")
JPEG_IMAGES_DIR = find_subdirectory(dataset_path, "JPEGImages")

# Fall-back: try Train subdirectories (INRIA has Train/ and Test/ splits)
if ANNOTATIONS_DIR is None:
    ANNOTATIONS_DIR = find_subdirectory(dataset_path, "annotations")
if JPEG_IMAGES_DIR is None:
    JPEG_IMAGES_DIR = find_subdirectory(dataset_path, "JPEGImages")

print(f"Annotations dir : {ANNOTATIONS_DIR}")
print(f"JPEGImages dir  : {JPEG_IMAGES_DIR}")

assert ANNOTATIONS_DIR is not None, "Could not find Annotations directory!"
assert JPEG_IMAGES_DIR is not None, "Could not find JPEGImages directory!"


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  CONFIGURATION                                                      ║
# ╚══════════════════════════════════════════════════════════════════════╝

CONFIG = {
    # Feature extraction method: "hog" or "cnn"
    "feature_method": "hog",

    # CNN model (used only if feature_method == "cnn")
    "cnn_model": "resnet50",

    # ROI target size (width, height) — must be 64×128 as per spec
    "target_size": (64, 128),

    # HOG parameters
    "hog_pixels_per_cell": (8, 8),
    "hog_cells_per_block": (2, 2),
    "hog_orientations": 9,

    # Negative sampling
    "neg_samples_per_image": 10,

    # Classifier
    "svm_C": 1.0,
    "test_split": 0.2,

    # Detection parameters
    "detection_scale": 1.05,
    "detection_stride": 8,
    "nms_iou_threshold": 0.3,
    "confidence_threshold": 0.5,

    # Hard Negative Mining
    "hard_negative_mining": True,
    "hnm_samples_per_image": 5,

    # Feature save prefix
    "feature_prefix": "pedestrian_hog",

    # Paths
    "annotations_dir": ANNOTATIONS_DIR,
    "images_dir": JPEG_IMAGES_DIR,
    "features_dir": os.path.join(REPO_DIR, "features"),
}

print("\n=== Configuration ===")
for k, v in CONFIG.items():
    print(f"  {k}: {v}")


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  BLOCK 2 — XML Parsing & ROI Extraction                            ║
# ╚══════════════════════════════════════════════════════════════════════╝

print("\n" + "=" * 60)
print("  BLOCK 2 — XML Parsing & ROI Extraction")
print("=" * 60)

# --- 2.1  Parse annotations ---
annotations = parse_annotations(CONFIG["annotations_dir"])
total_persons = sum(len(a["objects"]) for a in annotations)
print(f"\nParsed {len(annotations)} annotated images containing {total_persons} person bounding boxes.")

# --- 2.2  Show sample annotations ---
def visualize_annotations(annotations, image_dir, n=4):
    """Display a few annotated images with bounding boxes."""
    n = min(n, len(annotations))
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 6))
    if n == 1:
        axes = [axes]
    for ax, ann in zip(axes, annotations[:n]):
        img_path = Path(image_dir) / ann["filename"]
        # Try with extensions if needed
        if not img_path.exists():
            for ext in [".jpg", ".jpeg", ".png"]:
                if img_path.with_suffix(ext).exists():
                    img_path = img_path.with_suffix(ext)
                    break
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        for obj in ann["objects"]:
            xmin, ymin, xmax, ymax = obj["bbox"]
            rect = mpatches.Rectangle(
                (xmin, ymin), xmax - xmin, ymax - ymin,
                linewidth=2, edgecolor="lime", facecolor="none"
            )
            ax.add_patch(rect)
            ax.text(xmin, ymin - 5, obj["name"], color="lime", fontsize=10,
                    fontweight="bold", backgroundcolor="black")
        ax.set_title(ann["filename"], fontsize=9)
        ax.axis("off")
    plt.suptitle("Sample Annotated Images with Person Bounding Boxes", fontsize=14)
    plt.tight_layout()
    plt.show()

visualize_annotations(annotations, CONFIG["images_dir"])

# --- 2.3  Extract positive samples ---
print("\nExtracting positive samples (person crops) …")
pos_images, pos_labels = extract_positive_samples(
    annotations, CONFIG["images_dir"], target_size=CONFIG["target_size"]
)
print(f"  Positive samples: {len(pos_images)}")

# --- 2.4  Extract negative samples ---
print("Extracting negative samples (background crops) …")
neg_images, neg_labels = extract_negative_samples(
    annotations, CONFIG["images_dir"],
    target_size=CONFIG["target_size"],
    samples_per_image=CONFIG["neg_samples_per_image"],
    seed=42,
)
print(f"  Negative samples: {len(neg_images)}")

# --- 2.5  Visualize patches ---
def show_patches(pos, neg, n=5):
    """Display sample positive and negative patches side by side."""
    fig, axes = plt.subplots(2, n, figsize=(2.5 * n, 6))
    for i in range(n):
        if i < len(pos):
            axes[0, i].imshow(pos[i])
        axes[0, i].set_title("Person" if i < len(pos) else "")
        axes[0, i].axis("off")
        if i < len(neg):
            axes[1, i].imshow(neg[i])
        axes[1, i].set_title("Background" if i < len(neg) else "")
        axes[1, i].axis("off")
    plt.suptitle("Positive (top) vs Negative (bottom) Patches — 64×128", fontsize=13)
    plt.tight_layout()
    plt.show()

show_patches(pos_images, neg_images)

# --- 2.6  Combine into training arrays ---
X_patches = np.concatenate([pos_images, neg_images], axis=0)
y_labels = np.concatenate([pos_labels, neg_labels], axis=0)
print(f"\nCombined dataset: {X_patches.shape[0]} patches  "
      f"(Positive: {pos_labels.sum()}, Negative: {(y_labels == 0).sum()})")


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  BLOCK 3 — Feature Extraction                                      ║
# ╚══════════════════════════════════════════════════════════════════════╝

print("\n" + "=" * 60)
print("  BLOCK 3 — Feature Extraction")
print("=" * 60)

FEATURE_METHOD = CONFIG["feature_method"]   # "hog" or "cnn"

print(f"\nUsing feature method: {FEATURE_METHOD.upper()}")

if FEATURE_METHOD == "hog":
    X_features = hog_features(
        X_patches,
        pixels_per_cell=CONFIG["hog_pixels_per_cell"],
        cells_per_block=CONFIG["hog_cells_per_block"],
        orientations=CONFIG["hog_orientations"],
    )
    CONFIG["feature_prefix"] = "pedestrian_hog"
elif FEATURE_METHOD == "cnn":
    X_features = cnn_features(
        X_patches,
        model_name=CONFIG["cnn_model"],
        batch_size=32,
    )
    CONFIG["feature_prefix"] = f"pedestrian_{CONFIG['cnn_model']}"
else:
    raise ValueError(f"Unknown feature method: {FEATURE_METHOD}")

print(f"Feature matrix shape: {X_features.shape}")

# --- Save features to disk ---
save_features(X_features, y_labels, CONFIG["feature_prefix"], CONFIG["features_dir"])


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  BLOCK 4 — Classifier Training (LinearSVC + Hard Negative Mining)   ║
# ╚══════════════════════════════════════════════════════════════════════╝

print("\n" + "=" * 60)
print("  BLOCK 4 — Classifier Training")
print("=" * 60)

# --- 4.1  Load features (demonstrates I/O round-trip) ---
X_feat, y_feat = load_features(CONFIG["feature_prefix"], CONFIG["features_dir"])

# --- 4.2  Train / test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X_feat, y_feat, test_size=CONFIG["test_split"],
    random_state=42, stratify=y_feat,
)
print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")

# --- 4.3  Train LinearSVC ---
print(f"\nTraining LinearSVC (C={CONFIG['svm_C']}) …")
clf = LinearSVC(C=CONFIG["svm_C"], max_iter=10000)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("\n--- Classification Report (before HNM) ---")
print(classification_report(y_test, y_pred, target_names=["Background", "Person"]))

# Confusion matrix
fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred,
    display_labels=["Background", "Person"],
    cmap="Blues", ax=ax,
)
ax.set_title("Confusion Matrix (before Hard Negative Mining)")
plt.tight_layout()
plt.show()

# --- 4.4  Hard Negative Mining ---
if CONFIG["hard_negative_mining"]:
    print("\n--- Hard Negative Mining ---")
    print("Scanning negative images for false positives …")

    def extract_feature_fn(image):
        """Extract feature for a single 64×128 patch using the configured method."""
        if FEATURE_METHOD == "hog":
            return hog_feature_single(
                image,
                pixels_per_cell=CONFIG["hog_pixels_per_cell"],
                cells_per_block=CONFIG["hog_cells_per_block"],
                orientations=CONFIG["hog_orientations"],
            )
        else:
            return cnn_feature_single(image, model_name=CONFIG["cnn_model"])

    # Gather negative-only images (images that have no person annotations)
    annotated_filenames = set()
    for ann in annotations:
        annotated_filenames.add(ann["filename"])
        # Also try without extension
        annotated_filenames.add(Path(ann["filename"]).stem)

    all_imgs = sorted(
        p for p in Path(CONFIG["images_dir"]).rglob("*")
        if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")
    )
    neg_only_images = [
        p for p in all_imgs
        if p.name not in annotated_filenames and p.stem not in annotated_filenames
    ]

    print(f"  Found {len(neg_only_images)} images without person annotations.")

    tw, th = CONFIG["target_size"]
    stride = CONFIG["detection_stride"] * 2  # Larger stride for speed during mining
    hard_neg_features = []
    hard_neg_labels = []
    max_hard_per_image = CONFIG["hnm_samples_per_image"]

    for img_path in neg_only_images[:50]:  # Limit to 50 images for speed
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]
        if h < th or w < tw:
            continue

        count = 0
        for (x, y, window) in sliding_window(img_rgb, stride, (tw, th)):
            if window.shape[0] != th or window.shape[1] != tw:
                continue
            feat = extract_feature_fn(window)
            prediction = clf.predict(feat.reshape(1, -1))[0]
            if prediction == 1:  # False positive!
                hard_neg_features.append(feat)
                hard_neg_labels.append(0)
                count += 1
                if count >= max_hard_per_image:
                    break

    if hard_neg_features:
        hard_X = np.array(hard_neg_features, dtype=np.float32)
        hard_y = np.array(hard_neg_labels, dtype=np.int32)
        print(f"  Collected {len(hard_X)} hard negative samples.")

        # Augment training set and retrain
        X_train_aug = np.concatenate([X_train, hard_X], axis=0)
        y_train_aug = np.concatenate([y_train, hard_y], axis=0)

        print(f"  Retraining LinearSVC with {len(X_train_aug)} samples …")
        clf = LinearSVC(C=CONFIG["svm_C"], max_iter=10000)
        clf.fit(X_train_aug, y_train_aug)

        y_pred_hnm = clf.predict(X_test)
        print("\n--- Classification Report (after HNM) ---")
        print(classification_report(y_test, y_pred_hnm,
                                     target_names=["Background", "Person"]))

        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay.from_predictions(
            y_test, y_pred_hnm,
            display_labels=["Background", "Person"],
            cmap="Greens", ax=ax,
        )
        ax.set_title("Confusion Matrix (after Hard Negative Mining)")
        plt.tight_layout()
        plt.show()
    else:
        print("  No hard negatives found — model is already strong on negatives.")


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  BLOCK 5 — Object Detection Inference                              ║
# ╚══════════════════════════════════════════════════════════════════════╝

print("\n" + "=" * 60)
print("  BLOCK 5 — Object Detection Inference")
print("=" * 60)


def detect_pedestrians(image_path, model, feature_method="hog", config=None):
    """Detect pedestrians in an image using image pyramid + sliding window.

    Parameters
    ----------
    image_path : str
        Path to the input image.
    model : LinearSVC
        Trained classifier.
    feature_method : str
        "hog" or "cnn".
    config : dict
        Configuration dictionary.

    Returns
    -------
    detections : list of (x1, y1, x2, y2, score)
    image_rgb : np.ndarray  –  original image in RGB
    """
    if config is None:
        config = CONFIG

    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    tw, th = config["target_size"]
    stride = config["detection_stride"]
    scale_factor = config["detection_scale"]
    conf_thresh = config["confidence_threshold"]

    detections = []  # (x1, y1, x2, y2, score)

    for resized_img, current_scale in image_pyramid(img_rgb, scale=scale_factor, min_size=(tw, th)):
        for (x, y, window) in sliding_window(resized_img, stride, (tw, th)):
            if window.shape[0] != th or window.shape[1] != tw:
                continue

            # Extract features
            if feature_method == "hog":
                feat = hog_feature_single(
                    window,
                    pixels_per_cell=config["hog_pixels_per_cell"],
                    cells_per_block=config["hog_cells_per_block"],
                    orientations=config["hog_orientations"],
                )
            else:
                feat = cnn_feature_single(window, model_name=config["cnn_model"])

            # Predict
            score = model.decision_function(feat.reshape(1, -1))[0]

            if score > conf_thresh:
                # Project coordinates back to original image scale
                x1 = int(x * current_scale)
                y1 = int(y * current_scale)
                x2 = int((x + tw) * current_scale)
                y2 = int((y + th) * current_scale)
                detections.append((x1, y1, x2, y2, score))

    print(f"  Raw detections: {len(detections)}")
    return detections, img_rgb


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  BLOCK 6 — Post-processing (NMS) & Visualization                   ║
# ╚══════════════════════════════════════════════════════════════════════╝

print("\n" + "=" * 60)
print("  BLOCK 6 — NMS & Visualization")
print("=" * 60)


def detect_and_visualize(image_path, model, feature_method="hog", config=None):
    """Full pipeline: detect → NMS → draw bounding boxes → display."""
    if config is None:
        config = CONFIG

    detections, img_rgb = detect_pedestrians(image_path, model, feature_method, config)

    if len(detections) == 0:
        print("  No pedestrians detected.")
        plt.figure(figsize=(10, 8))
        plt.imshow(img_rgb)
        plt.title("No detections")
        plt.axis("off")
        plt.show()
        return

    # Separate boxes and scores
    boxes = np.array([(d[0], d[1], d[2], d[3]) for d in detections])
    scores = np.array([d[4] for d in detections])

    # Apply NMS
    keep = non_max_suppression(boxes, scores, iou_threshold=config["nms_iou_threshold"])
    final_boxes = boxes[keep]
    final_scores = scores[keep]
    print(f"  After NMS: {len(final_boxes)} detections")

    # Draw bounding boxes
    img_draw = img_rgb.copy()
    for i, (box, score) in enumerate(zip(final_boxes, final_scores)):
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(img_draw, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"Person {score:.2f}"
        cv2.putText(img_draw, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display
    plt.figure(figsize=(12, 9))
    plt.imshow(img_draw)
    plt.title(f"Pedestrian Detection — {len(final_boxes)} person(s) detected",
              fontsize=14)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


# --- Run detection on sample images ---

# Pick a few annotated images to test on
test_image_paths = []
for ann in annotations[:5]:
    img_path = Path(CONFIG["images_dir"]) / ann["filename"]
    if not img_path.exists():
        for ext in [".jpg", ".jpeg", ".png"]:
            if img_path.with_suffix(ext).exists():
                img_path = img_path.with_suffix(ext)
                break
    if img_path.exists():
        test_image_paths.append(str(img_path))

print(f"\nRunning detection on {len(test_image_paths)} test images …\n")

for img_path in test_image_paths:
    print(f"Processing: {Path(img_path).name}")
    detect_and_visualize(img_path, clf, FEATURE_METHOD, CONFIG)
    print()

print("=" * 60)
print("  Pipeline complete ✓")
print("=" * 60)
