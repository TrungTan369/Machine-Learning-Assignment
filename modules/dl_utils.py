"""
dl_utils.py - Deep learning helpers for Bai 3 (Image Data).

Two responsibilities:
  1. Pretrained feature extraction (ResNet50, VGG16, EfficientNetB0).
     The extractor is configurable: callers pick the model name, the
     image size, the pooling (avg / max / flatten) and the batch size.
  2. End-to-end transfer-learning helper for the deep-learning bonus
     pipeline (transfer learning + optional fine-tuning).

TensorFlow is imported lazily so that ml_utils-only callers do not pay
the import cost.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np


_MODEL_CACHE: dict = {}


# ---------------------------------------------------------------------------
# 1. Pretrained feature extraction
# ---------------------------------------------------------------------------

SUPPORTED_MODELS = ("resnet50", "vgg16", "efficientnetb0")


def _get_model_and_preprocess(model_name: str,
                              input_shape: tuple[int, int, int],
                              pooling: str):
    """Build (or return cached) feature extractor + preprocess function."""
    import tensorflow as tf
    from tensorflow.keras.applications import resnet50, vgg16, efficientnet

    cache_key = (model_name.lower(), input_shape, pooling)
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    name = model_name.lower()
    if name == "resnet50":
        base = resnet50.ResNet50(weights="imagenet", include_top=False,
                                 input_shape=input_shape)
        preprocess = resnet50.preprocess_input
    elif name == "vgg16":
        base = vgg16.VGG16(weights="imagenet", include_top=False,
                           input_shape=input_shape)
        preprocess = vgg16.preprocess_input
    elif name == "efficientnetb0":
        base = efficientnet.EfficientNetB0(weights="imagenet", include_top=False,
                                            input_shape=input_shape)
        preprocess = efficientnet.preprocess_input
    else:
        raise ValueError(
            f"Unsupported model {model_name!r}. Choose one of {SUPPORTED_MODELS}.")

    x = base.output
    if pooling == "avg":
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
    elif pooling == "max":
        x = tf.keras.layers.GlobalMaxPooling2D()(x)
    elif pooling == "flatten":
        x = tf.keras.layers.Flatten()(x)
    else:
        raise ValueError("pooling must be one of {'avg','max','flatten'}")

    model = tf.keras.Model(inputs=base.input, outputs=x)
    _MODEL_CACHE[cache_key] = (model, preprocess)
    return model, preprocess


def extract_features(images: np.ndarray,
                     model_name: str = "resnet50",
                     image_size: tuple[int, int] = (224, 224),
                     pooling: str = "avg",
                     batch_size: int = 32,
                     verbose: int = 1) -> np.ndarray:
    """Extract pretrained features from a stack of RGB images.

    Parameters
    ----------
    images : np.ndarray
        ``(N, H, W, 3)`` uint8 or float, any input H/W. Will be resized
        to *image_size* before going through the extractor.
    model_name : ``'resnet50'`` | ``'vgg16'`` | ``'efficientnetb0'``
    image_size : ``(W, H)`` for the network input
    pooling : ``'avg'`` | ``'max'`` | ``'flatten'``
    batch_size : forwarded to ``model.predict``

    Returns
    -------
    np.ndarray of shape ``(N, D)`` float32
    """
    import tensorflow as tf

    if images.size == 0:
        return np.empty((0, 0), dtype=np.float32)

    h, w = image_size[1], image_size[0]
    model, preprocess = _get_model_and_preprocess(
        model_name, (h, w, 3), pooling)

    if images.shape[1] != h or images.shape[2] != w:
        images_resized = tf.image.resize(images, (h, w)).numpy().astype(np.float32)
    else:
        images_resized = images.astype(np.float32, copy=True)

    processed = preprocess(images_resized)
    feats = model.predict(processed, batch_size=batch_size, verbose=verbose)
    return feats.astype(np.float32)


def feature_dim(model_name: str, image_size: tuple[int, int] = (224, 224),
                pooling: str = "avg") -> int:
    """Return the output dimensionality without running inference."""
    h, w = image_size[1], image_size[0]
    model, _ = _get_model_and_preprocess(model_name, (h, w, 3), pooling)
    return int(model.output_shape[-1])


# ---------------------------------------------------------------------------
# 2. End-to-end transfer learning (used for the deep-pipeline bonus)
# ---------------------------------------------------------------------------

def build_transfer_model(model_name: str = "vgg16",
                         image_size: tuple[int, int] = (224, 224),
                         num_classes: int = 2,
                         dropout: float = 0.3,
                         dense_units: int = 0):
    """Build a transfer-learning classifier with a frozen pretrained backbone.

    Returns
    -------
    model : tf.keras.Model  (compile separately)
    base_model : tf.keras.Model  (the frozen backbone, exposed so the
                                  caller can later un-freeze it for fine-tuning)
    """
    import tensorflow as tf
    from tensorflow.keras.applications import resnet50, vgg16, efficientnet

    h, w = image_size[1], image_size[0]
    name = model_name.lower()
    if name == "resnet50":
        base = resnet50.ResNet50(weights="imagenet", include_top=False,
                                 input_shape=(h, w, 3))
        preprocess = resnet50.preprocess_input
    elif name == "vgg16":
        base = vgg16.VGG16(weights="imagenet", include_top=False,
                           input_shape=(h, w, 3))
        preprocess = vgg16.preprocess_input
    elif name == "efficientnetb0":
        base = efficientnet.EfficientNetB0(weights="imagenet", include_top=False,
                                            input_shape=(h, w, 3))
        preprocess = efficientnet.preprocess_input
    else:
        raise ValueError(
            f"Unsupported model {model_name!r}. Choose one of {SUPPORTED_MODELS}.")

    base.trainable = False

    # Keras 3 (TF >= 2.16) disallows raw tf ops on a KerasTensor, so the
    # backbone-specific preprocess function has to live inside a Lambda layer.
    def _prep(t, _preprocess=preprocess):
        return _preprocess(tf.cast(t, tf.float32))

    inputs = tf.keras.Input(shape=(h, w, 3))
    x = tf.keras.layers.Lambda(_prep, output_shape=(h, w, 3),
                               name=f"{name}_preprocess")(inputs)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    if dropout > 0:
        x = tf.keras.layers.Dropout(dropout)(x)
    if dense_units > 0:
        x = tf.keras.layers.Dense(dense_units, activation="relu")(x)
    if num_classes <= 2:
        outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="prob")(x)
    else:
        outputs = tf.keras.layers.Dense(num_classes, activation="softmax",
                                        name="probs")(x)

    model = tf.keras.Model(inputs, outputs)
    return model, base


def unfreeze_top_layers(base_model, n_layers: int) -> int:
    """Unfreeze the top *n_layers* layers of *base_model*.

    Returns the number of trainable layers in the base after the change.
    """
    base_model.trainable = True
    if n_layers >= len(base_model.layers):
        return sum(1 for l in base_model.layers if l.trainable)
    for layer in base_model.layers[:-n_layers]:
        layer.trainable = False
    return n_layers


# ---------------------------------------------------------------------------
# 3. Feature I/O (mirror of ml_utils for convenience)
# ---------------------------------------------------------------------------

def save_features(features: np.ndarray, labels: np.ndarray, prefix: str,
                  out_dir: str | Path = "features") -> tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    x_path = out / f"{prefix}_X.npy"
    y_path = out / f"{prefix}_y.npy"
    np.save(str(x_path), features)
    np.save(str(y_path), labels)
    return x_path, y_path


def load_features(prefix: str, out_dir: str | Path = "features"
                  ) -> tuple[np.ndarray, np.ndarray]:
    out = Path(out_dir)
    X = np.load(str(out / f"{prefix}_X.npy"))
    y = np.load(str(out / f"{prefix}_y.npy"))
    return X, y
