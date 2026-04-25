from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications import resnet50, vgg16

def load_and_preprocess_data_tf(dataset_path, image_size=(224,224), batch_size=32, shuffle=False):
    ds = image_dataset_from_directory(str(dataset_path), labels='inferred', label_mode='int',
                                      image_size=image_size, batch_size=batch_size, shuffle=shuffle)
    class_names = ds.class_names
    imgs, lbls = [], []
    for b_x, b_y in ds:
        imgs.append(b_x.numpy()); lbls.append(b_y.numpy())
    if not imgs:
        return np.zeros((0,*image_size,3),dtype=np.float32), np.zeros((0,),dtype=np.int32), class_names
    return np.concatenate(imgs,0).astype(np.float32), np.concatenate(lbls,0).astype(np.int32), class_names

def _get_model_and_preprocess(model_name, input_shape, pooling='avg'):
    m = model_name.lower()
    if m == 'resnet50':
        base = resnet50.ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
        preprocess = resnet50.preprocess_input
    elif m == 'vgg16':
        base = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
        preprocess = vgg16.preprocess_input
    else:
        raise ValueError("model_name must be 'resnet50' or 'vgg16'")
    x = base.output
    if pooling == 'avg':
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
    model = tf.keras.Model(inputs=base.input, outputs=x)
    return model, preprocess

def extract_features_pretrained(X_images, model_name='resnet50', batch_size=32, pooling='avg', verbose=1):
    H,W = X_images.shape[1], X_images.shape[2]
    model, preprocess = _get_model_and_preprocess(model_name, (H,W,3), pooling)
    X_proc = preprocess(X_images.copy())
    feats = model.predict(X_proc, batch_size=batch_size, verbose=verbose)
    return feats

def save_features_to_disk(features, labels, prefix, out_dir='features'):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    x_path = out / f"{prefix}_X.npy"; y_path = out / f"{prefix}_y.npy"
    np.save(str(x_path), features); np.save(str(y_path), labels)
    return str(x_path), str(y_path)