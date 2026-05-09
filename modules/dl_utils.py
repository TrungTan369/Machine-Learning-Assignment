"""
dl_utils.py — Deep Learning utilities for the Pedestrian Detection pipeline.

Provides CNN-based feature extraction using pre-trained models (ResNet50, VGG16)
adapted for 64×128 ROI images.
"""

from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import resnet50, vgg16


# Cache for the loaded model so it isn't rebuilt on every call
_MODEL_CACHE = {}


def _get_model_and_preprocess(model_name, input_shape=(224, 224, 3), pooling='avg'):
    """Build a headless pre-trained model with global average pooling.

    Parameters
    ----------
    model_name : str  ``'resnet50'`` or ``'vgg16'``
    input_shape : tuple  e.g. ``(224, 224, 3)``
    pooling : str  ``'avg'`` for GlobalAveragePooling2D

    Returns
    -------
    model : tf.keras.Model
    preprocess_fn : callable
    """
    cache_key = (model_name, input_shape, pooling)
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    m = model_name.lower()
    if m == 'resnet50':
        base = resnet50.ResNet50(weights='imagenet', include_top=False,
                                 input_shape=input_shape)
        preprocess_fn = resnet50.preprocess_input
    elif m == 'vgg16':
        base = vgg16.VGG16(weights='imagenet', include_top=False,
                           input_shape=input_shape)
        preprocess_fn = vgg16.preprocess_input
    else:
        raise ValueError("model_name must be 'resnet50' or 'vgg16'")

    x = base.output
    if pooling == 'avg':
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
    model = tf.keras.Model(inputs=base.input, outputs=x)

    _MODEL_CACHE[cache_key] = (model, preprocess_fn)
    return model, preprocess_fn


def cnn_features(images, model_name='resnet50', batch_size=32):
    """Extract flattened CNN features from ROI images.

    Parameters
    ----------
    images : np.ndarray  ``(N, H, W, 3)`` — typically 128×64 ROIs (uint8, RGB)
    model_name : str  ``'resnet50'`` or ``'vgg16'``
    batch_size : int

    Returns
    -------
    np.ndarray  ``(N, D)`` — 1-D feature vectors (float32)
    """
    cnn_input_size = (224, 224)
    model, preprocess_fn = _get_model_and_preprocess(
        model_name, input_shape=(*cnn_input_size, 3), pooling='avg'
    )

    # Resize all ROIs to 224×224 for the pre-trained CNN
    resized = np.array([
        cv2.resize(img, cnn_input_size) for img in images
    ], dtype=np.float32)

    # Apply model-specific preprocessing
    processed = preprocess_fn(resized.copy())

    # Extract features
    feats = model.predict(processed, batch_size=batch_size, verbose=1)
    return feats.astype(np.float32)


def cnn_feature_single(image, model_name='resnet50'):
    """Extract CNN feature for a **single** RGB image.

    Parameters
    ----------
    image : np.ndarray  ``(H, W, 3)``
    model_name : str  ``'resnet50'`` or ``'vgg16'``

    Returns
    -------
    np.ndarray  ``(D,)`` — 1-D feature vector (float32)
    """
    feats = cnn_features(image[np.newaxis], model_name=model_name, batch_size=1)
    return feats[0]


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