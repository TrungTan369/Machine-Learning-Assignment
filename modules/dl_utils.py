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


def save_features(features, labels, filepath_prefix):
    """Save feature matrix and labels as ``.npy`` files.

    Creates ``<prefix>_X.npy`` and ``<prefix>_y.npy``.
    """
    out_dir = Path(filepath_prefix).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(f"{filepath_prefix}_X.npy", features)
    np.save(f"{filepath_prefix}_y.npy", labels)
    print(f"Saved features → {filepath_prefix}_X.npy  ({features.shape})")
    print(f"Saved labels   → {filepath_prefix}_y.npy  ({labels.shape})")


def load_features(filepath_prefix):
    """Load feature matrix and labels from ``.npy`` files.

    Returns ``(features, labels)`` numpy arrays.
    """
    X = np.load(f"{filepath_prefix}_X.npy")
    y = np.load(f"{filepath_prefix}_y.npy")
    return X, y