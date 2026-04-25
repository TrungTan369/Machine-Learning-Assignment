from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras import layers
from tensorflow.keras.applications import vgg16, resnet50

def load_and_preprocess_data(dataset_path, image_size=(224,224), batch_size=32, shuffle=False):
    """
    Load images from a directory structured as root/class_x/*.jpg
    Returns: X (np.float32, range depends on model preprocess), y (np.int32), class_names (list)
    """
    root = Path(dataset_path)
    if not root.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    ds = image_dataset_from_directory(
        str(root),
        labels='inferred',
        label_mode='int',
        image_size=image_size,
        batch_size=batch_size,
        shuffle=shuffle
    )
    class_names = ds.class_names

    imgs = []
    lbls = []
    for batch_img, batch_lbl in ds:
        imgs.append(batch_img.numpy())
        lbls.append(batch_lbl.numpy())
    if len(imgs) == 0:
        return np.zeros((0, *image_size, 3), dtype=np.float32), np.zeros((0,), dtype=np.int32), class_names

    X = np.concatenate(imgs, axis=0).astype(np.float32)
    y = np.concatenate(lbls, axis=0).astype(np.int32)
    return X, y, class_names

def _get_model_and_preprocess(model_name, input_shape, pooling='avg'):
    m = model_name.lower()
    if m == 'vgg16':
        base = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
        preprocess = vgg16.preprocess_input
    elif m in ('resnet50','resnet'):
        base = resnet50.ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
        preprocess = resnet50.preprocess_input
    else:
        raise ValueError("model_name must be 'vgg16' or 'resnet50'")

    x = base.output
    if pooling == 'avg':
        x = layers.GlobalAveragePooling2D()(x)
    elif pooling == 'max':
        x = layers.GlobalMaxPooling2D()(x)
    model = tf.keras.Model(inputs=base.input, outputs=x)
    return model, preprocess

def extract_features_pretrained(X_images, model_name='resnet50', batch_size=32, pooling='avg', verbose=1):
    """
    X_images: np.array (N,H,W,3) in uint8/float32
    Returns: features np.array (N, D)
    """
    if X_images.ndim != 4 or X_images.shape[-1] != 3:
        raise ValueError("X_images must be shape (N,H,W,3)")

    H, W = X_images.shape[1], X_images.shape[2]
    model, preprocess = _get_model_and_preprocess(model_name, (H, W, 3), pooling)
    X_proc = preprocess(X_images.copy())
    features = model.predict(X_proc, batch_size=batch_size, verbose=verbose)
    return features

def save_features_to_disk(features, labels, prefix, out_dir='features'):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    x_path = out / f"{prefix}_X.npy"
    y_path = out / f"{prefix}_y.npy"
    np.save(str(x_path), features)
    np.save(str(y_path), labels)
    return str(x_path), str(y_path)