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

The pipeline is configurable: callers pick the image size, the pretrained
feature extractor (see dl_utils), and the classifier. Helpers do not import
TensorFlow so they stay cheap to import in classical-only runs.
"""

from __future__ import annotations

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
