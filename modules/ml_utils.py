"""
ml_utils.py — Machine Learning utilities for the Pedestrian Detection pipeline.

Provides:
  - PASCAL VOC XML annotation parsing
  - Positive / Negative ROI extraction
  - HOG feature extraction (skimage)
  - Sliding window & image pyramid generators
  - Non-Maximum Suppression (NMS)
  - Feature I/O helpers
"""

import os
import random
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
from skimage import color, feature


# ---------------------------------------------------------------------------
# 1. Annotation parsing
# ---------------------------------------------------------------------------

def parse_annotations(annotation_dir):
    """Parse all PASCAL VOC XML files in *annotation_dir*.

    Returns
    -------
    list[dict]
        Each dict: ``{"filename": str, "objects": [{"name": str,
        "bbox": [xmin, ymin, xmax, ymax]}]}``.
        Only ``<name>person</name>`` objects are kept.
    """
    annotation_dir = Path(annotation_dir)
    results = []
    for xml_file in sorted(annotation_dir.glob("*.xml")):
        tree = ET.parse(str(xml_file))
        root = tree.getroot()

        # Filename from XML
        fname_el = root.find("filename")
        if fname_el is None:
            continue
        filename = fname_el.text

        objects = []
        for obj in root.iter("object"):
            name_el = obj.find("name")
            if name_el is None or name_el.text.strip().lower() != "person":
                continue
            bbox_el = obj.find("bndbox")
            if bbox_el is None:
                continue
            xmin = int(float(bbox_el.find("xmin").text))
            ymin = int(float(bbox_el.find("ymin").text))
            xmax = int(float(bbox_el.find("xmax").text))
            ymax = int(float(bbox_el.find("ymax").text))
            objects.append({"name": "person", "bbox": [xmin, ymin, xmax, ymax]})

        results.append({"filename": filename, "objects": objects})
    return results


# ---------------------------------------------------------------------------
# 2. ROI extraction
# ---------------------------------------------------------------------------

def compute_iou(box_a, box_b):
    """Compute Intersection-over-Union for two boxes ``[x1, y1, x2, y2]``."""
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def extract_positive_samples(annotation_data, image_dir, target_size=(64, 128)):
    """Crop each person bounding box and resize to *target_size* (W, H).

    Parameters
    ----------
    annotation_data : list[dict]
        Output of :func:`parse_annotations`.
    image_dir : str | Path
        Directory containing JPEG images.
    target_size : tuple
        ``(width, height)`` — default ``(64, 128)``.

    Returns
    -------
    images : np.ndarray of shape ``(N, H, W, 3)``
    labels : np.ndarray of shape ``(N,)`` filled with 1
    """
    image_dir = Path(image_dir)
    rois = []
    for entry in annotation_data:
        if not entry["objects"]:
            continue
        img_path = image_dir / entry["filename"]
        if not img_path.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        for obj in entry["objects"]:
            x1, y1, x2, y2 = obj["bbox"]
            # Clamp to image boundaries
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img.shape[1], x2)
            y2 = min(img.shape[0], y2)
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            crop = cv2.resize(crop, target_size)
            rois.append(crop)

    images = np.array(rois, dtype=np.uint8)
    labels = np.ones(len(rois), dtype=np.int32)
    return images, labels


def extract_negative_samples(annotation_data, image_dir,
                             target_size=(64, 128), samples_per_image=10,
                             seed=42):
    """Extract random negative patches that have IoU = 0 with all person boxes.

    For images with person annotations, random windows are sampled from
    regions that do **not** overlap any annotated person.  For images with
    no annotations at all, any random window is a valid negative.

    Parameters
    ----------
    annotation_data : list[dict]
        Output of :func:`parse_annotations`.
    image_dir : str | Path
        Directory containing JPEG images.
    target_size : tuple
        ``(width, height)`` — default ``(64, 128)``.
    samples_per_image : int
        How many negative crops to attempt per image.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    images : np.ndarray  ``(N, H, W, 3)``
    labels : np.ndarray  ``(N,)`` filled with 0
    """
    rng = random.Random(seed)
    image_dir = Path(image_dir)
    w_target, h_target = target_size
    rois = []

    for entry in annotation_data:
        img_path = image_dir / entry["filename"]
        if not img_path.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h_img, w_img = img.shape[:2]

        if h_img < h_target or w_img < w_target:
            continue

        person_boxes = [o["bbox"] for o in entry["objects"]]

        attempts = 0
        collected = 0
        max_attempts = samples_per_image * 10
        while collected < samples_per_image and attempts < max_attempts:
            attempts += 1
            # Random scale factor to get patches of varying sizes
            scale = rng.uniform(1.0, 2.0)
            crop_w = int(w_target * scale)
            crop_h = int(h_target * scale)
            if crop_w > w_img or crop_h > h_img:
                crop_w = w_target
                crop_h = h_target

            x1 = rng.randint(0, w_img - crop_w)
            y1 = rng.randint(0, h_img - crop_h)
            x2 = x1 + crop_w
            y2 = y1 + crop_h

            # Check IoU with all person boxes
            overlaps = False
            for pb in person_boxes:
                if compute_iou([x1, y1, x2, y2], pb) > 0:
                    overlaps = True
                    break
            if overlaps:
                continue

            crop = img[y1:y2, x1:x2]
            crop = cv2.resize(crop, target_size)
            rois.append(crop)
            collected += 1

    images = np.array(rois, dtype=np.uint8)
    labels = np.zeros(len(rois), dtype=np.int32)
    return images, labels


# ---------------------------------------------------------------------------
# 3. HOG feature extraction
# ---------------------------------------------------------------------------

def hog_features(images, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                 orientations=9):
    """Compute HOG descriptors for an array of 64×128 RGB images.

    Parameters
    ----------
    images : np.ndarray  ``(N, 128, 64, 3)``
    pixels_per_cell, cells_per_block, orientations : HOG params

    Returns
    -------
    np.ndarray  ``(N, D)`` — feature matrix (float32)
    """
    feats = []
    for img in images:
        gray = color.rgb2gray(img)  # float64 in [0, 1]
        h = feature.hog(gray,
                        orientations=orientations,
                        pixels_per_cell=pixels_per_cell,
                        cells_per_block=cells_per_block,
                        feature_vector=True)
        feats.append(h)
    return np.array(feats, dtype=np.float32)


def hog_feature_single(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                        orientations=9):
    """Compute HOG descriptor for a **single** RGB image.

    Parameters
    ----------
    image : np.ndarray  ``(H, W, 3)``

    Returns
    -------
    np.ndarray  ``(D,)`` — 1-D feature vector (float32)
    """
    gray = color.rgb2gray(image)
    h = feature.hog(gray,
                    orientations=orientations,
                    pixels_per_cell=pixels_per_cell,
                    cells_per_block=cells_per_block,
                    feature_vector=True)
    return np.array(h, dtype=np.float32)


# ---------------------------------------------------------------------------
# 4. Sliding window & image pyramid
# ---------------------------------------------------------------------------

def sliding_window(image, step_size, window_size):
    """Yield ``(x, y, window)`` over *image*.

    Parameters
    ----------
    image : np.ndarray  ``(H, W, ...)``
    step_size : int  stride in pixels
    window_size : tuple  ``(w, h)``
    """
    w_win, h_win = window_size
    for y in range(0, image.shape[0] - h_win + 1, step_size):
        for x in range(0, image.shape[1] - w_win + 1, step_size):
            yield (x, y, image[y:y + h_win, x:x + w_win])


def image_pyramid(image, scale=1.05, min_size=(64, 128)):
    """Yield ``(resized_image, cumulative_scale)`` for progressively
    down-scaled copies of *image*.

    The first yield is the original image at scale 1.0.

    Parameters
    ----------
    image : np.ndarray
    scale : float  down-scale factor per step (> 1.0)
    min_size : tuple  ``(min_width, min_height)``
    """
    current_scale = 1.0
    yield image, current_scale
    while True:
        current_scale *= scale
        w = int(image.shape[1] / current_scale)
        h = int(image.shape[0] / current_scale)
        if w < min_size[0] or h < min_size[1]:
            break
        resized = cv2.resize(image, (w, h))
        yield resized, current_scale


# ---------------------------------------------------------------------------
# 5. Non-Maximum Suppression
# ---------------------------------------------------------------------------

def non_max_suppression(boxes, scores, iou_threshold=0.3):
    """Greedy NMS.

    Parameters
    ----------
    boxes : np.ndarray  ``(N, 4)`` — ``[x1, y1, x2, y2]``
    scores : np.ndarray  ``(N,)``
    iou_threshold : float

    Returns
    -------
    keep : np.ndarray of int — indices of kept boxes
    """
    if len(boxes) == 0:
        return np.empty((0,), dtype=int)

    boxes = np.array(boxes, dtype=np.float32)
    scores = np.array(scores, dtype=np.float32)

    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break

        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_rest = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        iou = inter / (area_i + area_rest - inter + 1e-6)

        inds = np.where(iou <= iou_threshold)[0]
        order = rest[inds]

    return np.array(keep, dtype=int)


# ---------------------------------------------------------------------------
# 6. Feature I/O
# ---------------------------------------------------------------------------

def save_features(features, labels, prefix, out_dir="features"):
    """Save feature matrix and labels as ``.npy`` files.

    Creates ``<out_dir>/<prefix>_X.npy`` and ``<out_dir>/<prefix>_y.npy``.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    x_path = out / f"{prefix}_X.npy"
    y_path = out / f"{prefix}_y.npy"
    np.save(str(x_path), features)
    np.save(str(y_path), labels)
    print(f"Saved features → {x_path}  ({features.shape})")
    print(f"Saved labels   → {y_path}  ({labels.shape})")


def load_features(prefix, out_dir="features"):
    """Load feature matrix and labels from ``.npy`` files.

    Returns ``(features, labels)`` numpy arrays.
    """
    out = Path(out_dir)
    X = np.load(str(out / f"{prefix}_X.npy"))
    y = np.load(str(out / f"{prefix}_y.npy"))
    return X, y