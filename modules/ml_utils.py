"""
ml_utils.py - Image classification helpers for Bai 3 (Image Data).

Functions are grouped into:
  1. Dataset discovery & loading      (find_class_dirs, load_image_paths,
                                       load_and_resize_images, build_dataset)
  2. EDA                              (image_size_stats, channel_stats,
                                       label_distribution, plot_sample_grid,
                                       plot_label_distribution,
                                       plot_image_size_hist,
                                       plot_channel_means)
  3. Classifier comparison            (build_classifier, evaluate_classifier,
                                       compare_classifiers)
  4. Feature I/O                      (save_features, load_features)
  5. PASCAL VOC binary dataset        (find_voc_splits, parse_voc_annotations,
                                       extract_positive_samples,
                                       extract_negative_samples,
                                       build_voc_binary_dataset)

The pipeline is configurable: callers pick the image size, the pretrained
feature extractor (see dl_utils), and the classifier. Helpers do not import
TensorFlow so they stay cheap to import in classical-only runs.
"""

from __future__ import annotations

import random
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".ppm", ".pgm", ".tif", ".tiff"}


# ---------------------------------------------------------------------------
# 1. Dataset discovery & loading
# ---------------------------------------------------------------------------

def find_class_dirs(root: str | Path,
                    candidate_names: Sequence[str] = ("pos", "neg")) -> dict[str, Path]:
    """Walk *root* and return ``{class_name: directory}`` for every directory
    whose name matches *candidate_names* (case-insensitive).

    Useful for INRIA Person where pos/neg folders may live several levels
    deep (e.g. ``INRIAPerson/Train/pos``).
    """
    root = Path(root)
    matches: dict[str, list[Path]] = {name: [] for name in candidate_names}
    for d in root.rglob("*"):
        if not d.is_dir():
            continue
        name_lower = d.name.lower()
        for cand in candidate_names:
            if name_lower == cand.lower():
                matches[cand].append(d)

    out: dict[str, Path] = {}
    for cand, dirs in matches.items():
        if not dirs:
            continue
        # Prefer the shallowest directory (closest to root) for each class
        out[cand] = sorted(dirs, key=lambda p: len(p.parts))[0]
    return out


def load_image_paths(directory: str | Path) -> list[Path]:
    """Return image file paths inside *directory* (non-recursive)."""
    directory = Path(directory)
    paths: list[Path] = []
    for p in sorted(directory.iterdir()):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            paths.append(p)
    return paths


def load_image_paths_recursive(directory: str | Path) -> list[Path]:
    """Return image file paths under *directory* and any nested folder."""
    directory = Path(directory)
    paths: list[Path] = []
    for p in sorted(directory.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            paths.append(p)
    return paths


def load_and_resize_images(paths: Iterable[str | Path],
                           image_size: tuple[int, int] = (224, 224),
                           dtype=np.uint8) -> np.ndarray:
    """Load each image, convert to RGB, resize to *image_size* (W, H).

    Returns
    -------
    np.ndarray of shape ``(N, H, W, 3)`` in *dtype*.
    """
    out = []
    for p in paths:
        img = Image.open(str(p)).convert("RGB")
        img = img.resize(image_size, Image.BILINEAR)
        out.append(np.array(img, dtype=dtype))
    if not out:
        return np.empty((0, image_size[1], image_size[0], 3), dtype=dtype)
    return np.stack(out, axis=0)


def build_dataset(class_dirs: dict[str, Path],
                  image_size: tuple[int, int] = (224, 224),
                  recursive: bool = False,
                  max_per_class: int | None = None,
                  shuffle: bool = True,
                  seed: int = 42
                  ) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load images from each class directory into one (X, y, class_names).

    Parameters
    ----------
    class_dirs : dict
        ``{class_name: directory}``. Order of insertion is preserved and
        used to assign integer labels (0, 1, ...).
    image_size : (W, H)
    recursive : bool
        If True, search subdirectories of each class folder.
    max_per_class : int, optional
        Cap the number of images per class (useful for fast smoke runs).
    shuffle : bool
        Shuffle the assembled dataset.
    seed : int

    Returns
    -------
    X : np.ndarray ``(N, H, W, 3)`` uint8
    y : np.ndarray ``(N,)``    int64
    class_names : list[str]    ordered same as label ids
    """
    rng = np.random.default_rng(seed)
    class_names: list[str] = []
    X_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []

    for label, (cls, directory) in enumerate(class_dirs.items()):
        loader = load_image_paths_recursive if recursive else load_image_paths
        paths = loader(directory)
        if max_per_class is not None and len(paths) > max_per_class:
            idx = rng.permutation(len(paths))[:max_per_class]
            paths = [paths[i] for i in idx]
        if not paths:
            continue
        imgs = load_and_resize_images(paths, image_size=image_size)
        X_parts.append(imgs)
        y_parts.append(np.full(len(imgs), label, dtype=np.int64))
        class_names.append(cls)

    X = np.concatenate(X_parts, axis=0) if X_parts else np.empty(
        (0, image_size[1], image_size[0], 3), dtype=np.uint8)
    y = np.concatenate(y_parts, axis=0) if y_parts else np.empty((0,), dtype=np.int64)

    if shuffle and len(X) > 0:
        order = rng.permutation(len(X))
        X = X[order]
        y = y[order]

    return X, y, class_names


# ---------------------------------------------------------------------------
# 2. EDA
# ---------------------------------------------------------------------------

def image_size_stats(paths: Iterable[str | Path]) -> dict:
    """Compute width / height / channel statistics over a list of images.

    Reads only image headers (Pillow lazy-load) so it is fast.
    """
    widths: list[int] = []
    heights: list[int] = []
    modes: list[str] = []
    for p in paths:
        with Image.open(str(p)) as img:
            widths.append(img.width)
            heights.append(img.height)
            modes.append(img.mode)
    if not widths:
        return {"count": 0}
    widths_a = np.asarray(widths)
    heights_a = np.asarray(heights)
    return {
        "count": len(widths),
        "width_min": int(widths_a.min()),
        "width_max": int(widths_a.max()),
        "width_mean": float(widths_a.mean()),
        "height_min": int(heights_a.min()),
        "height_max": int(heights_a.max()),
        "height_mean": float(heights_a.mean()),
        "modes": dict(Counter(modes)),
        "widths": widths_a,
        "heights": heights_a,
    }


def channel_stats(images: np.ndarray) -> dict:
    """Per-channel mean and std for an ``(N, H, W, 3)`` uint8 array."""
    if images.size == 0:
        return {"mean": np.zeros(3), "std": np.zeros(3)}
    arr = images.astype(np.float32) / 255.0
    return {
        "mean": arr.mean(axis=(0, 1, 2)),
        "std": arr.std(axis=(0, 1, 2)),
    }


def label_distribution(y: np.ndarray, class_names: Sequence[str]) -> dict[str, int]:
    """Return ``{class_name: count}`` ordered by *class_names*."""
    counts = Counter(int(v) for v in y)
    return {name: int(counts.get(i, 0)) for i, name in enumerate(class_names)}


def plot_sample_grid(images: np.ndarray, y: np.ndarray, class_names: Sequence[str],
                     n_per_class: int = 4, figsize=(12, 6), title: str | None = None):
    """Plot a grid with *n_per_class* random samples per class."""
    import matplotlib.pyplot as plt

    n_classes = len(class_names)
    fig, axes = plt.subplots(n_classes, n_per_class,
                             figsize=figsize, squeeze=False)
    rng = np.random.default_rng(0)
    for r, cls in enumerate(class_names):
        idx = np.where(y == r)[0]
        pick = rng.choice(idx, size=min(n_per_class, len(idx)), replace=False)
        for c, i in enumerate(pick):
            axes[r, c].imshow(images[i])
            axes[r, c].axis("off")
            if c == 0:
                axes[r, c].set_ylabel(cls, fontsize=12)
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    return fig


def plot_label_distribution(y: np.ndarray, class_names: Sequence[str],
                            figsize=(6, 4)):
    """Bar chart of class counts."""
    import matplotlib.pyplot as plt

    counts = label_distribution(y, class_names)
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(list(counts.keys()), list(counts.values()),
           color=["tab:green", "tab:red", "tab:blue", "tab:orange"])
    ax.set_ylabel("# images")
    ax.set_title("Label distribution")
    for i, v in enumerate(counts.values()):
        ax.text(i, v, str(v), ha="center", va="bottom")
    plt.tight_layout()
    return fig


def plot_image_size_hist(stats: dict, figsize=(10, 4)):
    """Histogram of widths and heights from :func:`image_size_stats`."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    axes[0].hist(stats["widths"], bins=30, color="tab:blue")
    axes[0].set_title(f"Widths (mean={stats['width_mean']:.0f})")
    axes[0].set_xlabel("pixels")
    axes[1].hist(stats["heights"], bins=30, color="tab:orange")
    axes[1].set_title(f"Heights (mean={stats['height_mean']:.0f})")
    axes[1].set_xlabel("pixels")
    plt.tight_layout()
    return fig


def plot_channel_means(stats: dict, figsize=(5, 4)):
    """Bar plot of RGB channel means produced by :func:`channel_stats`."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(["R", "G", "B"], stats["mean"],
           color=["tab:red", "tab:green", "tab:blue"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean intensity (0-1)")
    ax.set_title("Per-channel mean")
    for i, v in enumerate(stats["mean"]):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 3. Classifier comparison
# ---------------------------------------------------------------------------

def build_classifier(name: str, **kwargs):
    """Factory for the classifiers compared in the report.

    Supported names: ``'logreg'``, ``'svm_linear'``, ``'svm_rbf'``,
    ``'random_forest'``.
    """
    name = name.lower()
    if name == "logreg":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(max_iter=kwargs.pop("max_iter", 2000),
                                  **kwargs)
    if name == "svm_linear":
        from sklearn.svm import LinearSVC
        return LinearSVC(C=kwargs.pop("C", 1.0),
                         max_iter=kwargs.pop("max_iter", 5000),
                         **kwargs)
    if name == "svm_rbf":
        from sklearn.svm import SVC
        return SVC(kernel="rbf",
                   C=kwargs.pop("C", 1.0),
                   probability=kwargs.pop("probability", False),
                   **kwargs)
    if name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=kwargs.pop("n_estimators", 200),
                                      n_jobs=kwargs.pop("n_jobs", -1),
                                      random_state=kwargs.pop("random_state", 42),
                                      **kwargs)
    raise ValueError(f"Unknown classifier name: {name}")


def evaluate_classifier(clf, X_train, y_train, X_test, y_test) -> dict:
    """Train *clf* and return a dict of test-set metrics + predictions."""
    from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                                 confusion_matrix)
    import time

    t0 = time.time()
    clf.fit(X_train, y_train)
    fit_time = time.time() - t0

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0)
    return {
        "accuracy": float(acc),
        "precision_macro": float(prec),
        "recall_macro": float(rec),
        "f1_macro": float(f1),
        "fit_time_sec": float(fit_time),
        "y_pred": y_pred,
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }


def compare_classifiers(features: dict[str, np.ndarray],
                        y: np.ndarray,
                        classifier_names: Sequence[str],
                        test_size: float = 0.2,
                        random_state: int = 42) -> "list[dict]":
    """Train every (feature, classifier) combination on the same split.

    Parameters
    ----------
    features : dict
        ``{feature_name: X}`` where each X has shape ``(N, D)`` and the
        same N as *y*.
    y : np.ndarray  ``(N,)``
    classifier_names : sequence of str
        Names accepted by :func:`build_classifier`.
    test_size, random_state : passed to ``train_test_split``.

    Returns
    -------
    list[dict] : one row per (feature, classifier) with metrics.
    """
    from sklearn.model_selection import train_test_split

    rows = []
    for feat_name, X in features.items():
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y)
        for clf_name in classifier_names:
            clf = build_classifier(clf_name)
            metrics = evaluate_classifier(clf, X_train, y_train, X_test, y_test)
            metrics.pop("y_pred", None)  # keep table compact
            metrics.pop("confusion_matrix", None)
            rows.append({
                "features": feat_name,
                "classifier": clf_name,
                **metrics,
            })
    return rows


# ---------------------------------------------------------------------------
# 4. Feature I/O
# ---------------------------------------------------------------------------

def save_features(features: np.ndarray, labels: np.ndarray, prefix: str,
                  out_dir: str | Path = "features") -> tuple[Path, Path]:
    """Save *features* and *labels* as ``<out_dir>/<prefix>_X.npy`` /
    ``<prefix>_y.npy``. Returns the two file paths.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    x_path = out / f"{prefix}_X.npy"
    y_path = out / f"{prefix}_y.npy"
    np.save(str(x_path), features)
    np.save(str(y_path), labels)
    return x_path, y_path


def load_features(prefix: str, out_dir: str | Path = "features"
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Load features and labels saved by :func:`save_features`."""
    out = Path(out_dir)
    X = np.load(str(out / f"{prefix}_X.npy"))
    y = np.load(str(out / f"{prefix}_y.npy"))
    return X, y


# ---------------------------------------------------------------------------
# 5. PASCAL VOC binary dataset (person / no-person)
# ---------------------------------------------------------------------------

def find_voc_splits(root: str | Path,
                    image_dir_names: Sequence[str] = ("JPEGImages", "Images", "pos"),
                    annot_dir_names: Sequence[str] = ("Annotations",)
                    ) -> list[dict]:
    """Find every (image_dir, annotation_dir) pair under *root*.

    A "split" is a parent directory that has both an image folder and a
    sibling annotation folder. Returns a list of
    ``{"name": <parent>, "image_dir": Path, "annotation_dir": Path}``.
    Useful for INRIA Person where Train/ and Test/ each ship JPEGImages/
    and Annotations/ side-by-side.
    """
    root = Path(root)
    img_set = {n.lower() for n in image_dir_names}
    ann_set = {n.lower() for n in annot_dir_names}

    splits: list[dict] = []
    for parent in sorted({p.parent for p in root.rglob("*") if p.is_dir()}):
        children = {c.name.lower(): c for c in parent.iterdir() if c.is_dir()}
        img_dir = next((children[k] for k in children if k in img_set), None)
        ann_dir = next((children[k] for k in children if k in ann_set), None)
        if img_dir is not None and ann_dir is not None:
            splits.append({
                "name": parent.name,
                "image_dir": img_dir,
                "annotation_dir": ann_dir,
            })
    return splits


def _compute_iou(a: list[int], b: list[int]) -> float:
    xa = max(a[0], b[0]); ya = max(a[1], b[1])
    xb = min(a[2], b[2]); yb = min(a[3], b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def parse_voc_annotations(annotation_dir: str | Path,
                          target_label: str = "person") -> list[dict]:
    """Parse every PASCAL VOC ``*.xml`` file in *annotation_dir*.

    Only objects whose ``<name>`` equals *target_label* (case-insensitive)
    are kept. Returns a list of dicts:
    ``{"filename": str, "objects": [{"name": str, "bbox": [x1,y1,x2,y2]}]}``.
    """
    annotation_dir = Path(annotation_dir)
    target_lower = target_label.lower()
    out: list[dict] = []
    for xml_file in sorted(annotation_dir.glob("*.xml")):
        tree = ET.parse(str(xml_file))
        root = tree.getroot()
        fname_el = root.find("filename")
        if fname_el is None or not fname_el.text:
            continue
        objects: list[dict] = []
        for obj in root.iter("object"):
            name_el = obj.find("name")
            bbox_el = obj.find("bndbox")
            if (name_el is None or bbox_el is None
                or name_el.text is None
                or name_el.text.strip().lower() != target_lower):
                continue
            try:
                xmin = int(float(bbox_el.findtext("xmin", "0")))
                ymin = int(float(bbox_el.findtext("ymin", "0")))
                xmax = int(float(bbox_el.findtext("xmax", "0")))
                ymax = int(float(bbox_el.findtext("ymax", "0")))
            except ValueError:
                continue
            if xmax > xmin and ymax > ymin:
                objects.append({"name": target_lower,
                                "bbox": [xmin, ymin, xmax, ymax]})
        out.append({"filename": fname_el.text, "objects": objects})
    return out


def _open_rgb(path: Path) -> Image.Image | None:
    try:
        return Image.open(str(path)).convert("RGB")
    except (FileNotFoundError, OSError):
        return None


def _resolve_image_path(image_dir: Path, filename: str) -> Path | None:
    """Try to find *filename* under *image_dir*, falling back to other extensions."""
    direct = image_dir / filename
    if direct.exists():
        return direct
    stem = Path(filename).stem
    for ext in (".png", ".jpg", ".jpeg", ".bmp", ".ppm"):
        candidate = image_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def extract_positive_samples(annotation_data: list[dict],
                             image_dir: str | Path,
                             target_size: tuple[int, int] = (64, 128)
                             ) -> tuple[np.ndarray, np.ndarray]:
    """Crop each ``person`` bounding box and resize to *target_size* (W, H).

    Returns ``(images, labels)`` where ``labels`` is an int array of 1s.
    """
    image_dir = Path(image_dir)
    rois: list[np.ndarray] = []
    for entry in annotation_data:
        if not entry["objects"]:
            continue
        img_path = _resolve_image_path(image_dir, entry["filename"])
        if img_path is None:
            continue
        img = _open_rgb(img_path)
        if img is None:
            continue
        w_img, h_img = img.size
        for obj in entry["objects"]:
            x1, y1, x2, y2 = obj["bbox"]
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(w_img, x2); y2 = min(h_img, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = img.crop((x1, y1, x2, y2)).resize(target_size, Image.BILINEAR)
            rois.append(np.array(crop, dtype=np.uint8))

    if not rois:
        return (np.empty((0, target_size[1], target_size[0], 3), dtype=np.uint8),
                np.empty((0,), dtype=np.int64))
    images = np.stack(rois, axis=0)
    labels = np.ones(len(rois), dtype=np.int64)
    return images, labels


def extract_negative_samples(annotation_data: list[dict],
                             image_dir: str | Path,
                             target_size: tuple[int, int] = (64, 128),
                             samples_per_image: int = 5,
                             seed: int = 42
                             ) -> tuple[np.ndarray, np.ndarray]:
    """Random patches whose IoU with every ``person`` bbox is 0.

    For images with no person annotations, any random window is accepted.
    """
    rng = random.Random(seed)
    image_dir = Path(image_dir)
    w_target, h_target = target_size
    rois: list[np.ndarray] = []

    for entry in annotation_data:
        img_path = _resolve_image_path(image_dir, entry["filename"])
        if img_path is None:
            continue
        img = _open_rgb(img_path)
        if img is None:
            continue
        w_img, h_img = img.size
        if w_img < w_target or h_img < h_target:
            continue
        boxes = [o["bbox"] for o in entry["objects"]]
        max_attempts = samples_per_image * 10
        collected = 0
        attempts = 0
        while collected < samples_per_image and attempts < max_attempts:
            attempts += 1
            scale = rng.uniform(1.0, 2.0)
            crop_w = int(w_target * scale)
            crop_h = int(h_target * scale)
            if crop_w > w_img or crop_h > h_img:
                crop_w, crop_h = w_target, h_target
            x1 = rng.randint(0, w_img - crop_w)
            y1 = rng.randint(0, h_img - crop_h)
            x2 = x1 + crop_w
            y2 = y1 + crop_h
            if any(_compute_iou([x1, y1, x2, y2], b) > 0 for b in boxes):
                continue
            crop = img.crop((x1, y1, x2, y2)).resize(target_size, Image.BILINEAR)
            rois.append(np.array(crop, dtype=np.uint8))
            collected += 1

    if not rois:
        return (np.empty((0, target_size[1], target_size[0], 3), dtype=np.uint8),
                np.empty((0,), dtype=np.int64))
    images = np.stack(rois, axis=0)
    labels = np.zeros(len(rois), dtype=np.int64)
    return images, labels


def build_voc_binary_dataset(splits: Sequence[dict],
                             target_size: tuple[int, int] = (224, 224),
                             samples_per_image: int = 5,
                             max_per_class: int | None = None,
                             shuffle: bool = True,
                             seed: int = 42,
                             target_label: str = "person",
                             verbose: bool = True
                             ) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build a (X, y, class_names) binary dataset from VOC-style splits.

    Each split contributes positive crops (person bboxes) and an equal
    *budget* of random negative crops drawn from the same image set.

    Class names are ``["neg", "pos"]`` so that label 0 = neg and 1 = pos
    (matches the convention used by extract_*_samples).

    Parameters
    ----------
    splits : output of :func:`find_voc_splits`
    target_size : (W, H) — final crop size before downstream resize
    samples_per_image : negatives sampled per image
    max_per_class : optional cap (applied after concatenating splits)
    shuffle, seed : reproducibility
    target_label : VOC ``<name>`` to treat as positive (default ``person``)
    """
    rng = np.random.default_rng(seed)
    pos_X, neg_X = [], []
    for sp in splits:
        annot = parse_voc_annotations(sp["annotation_dir"], target_label=target_label)
        if verbose:
            n_obj = sum(len(e["objects"]) for e in annot)
            print(f"  split {sp['name']}: {len(annot)} annotated images, "
                  f"{n_obj} {target_label} bboxes")
        Xp, _ = extract_positive_samples(annot, sp["image_dir"], target_size=target_size)
        Xn, _ = extract_negative_samples(annot, sp["image_dir"],
                                         target_size=target_size,
                                         samples_per_image=samples_per_image,
                                         seed=seed)
        if verbose:
            print(f"     -> pos crops={len(Xp)}, neg crops={len(Xn)}")
        pos_X.append(Xp); neg_X.append(Xn)

    Xp = np.concatenate(pos_X, axis=0) if pos_X else np.empty(
        (0, target_size[1], target_size[0], 3), dtype=np.uint8)
    Xn = np.concatenate(neg_X, axis=0) if neg_X else np.empty(
        (0, target_size[1], target_size[0], 3), dtype=np.uint8)

    if max_per_class is not None:
        if len(Xp) > max_per_class:
            idx = rng.permutation(len(Xp))[:max_per_class]
            Xp = Xp[idx]
        if len(Xn) > max_per_class:
            idx = rng.permutation(len(Xn))[:max_per_class]
            Xn = Xn[idx]

    X = np.concatenate([Xn, Xp], axis=0)
    y = np.concatenate([np.zeros(len(Xn), dtype=np.int64),
                        np.ones(len(Xp), dtype=np.int64)], axis=0)
    class_names = ["neg", "pos"]

    if shuffle and len(X) > 0:
        order = rng.permutation(len(X))
        X = X[order]
        y = y[order]
    return X, y, class_names
