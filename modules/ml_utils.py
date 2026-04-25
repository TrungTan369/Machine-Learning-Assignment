import os
from pathlib import Path
import numpy as np
from PIL import Image
from skimage import color, transform, feature

def load_image_paths(root_dir):
    root = Path(root_dir)
    paths = [p for p in root.rglob('*') if p.suffix.lower() in ('.jpg','.jpeg','.png')]
    return sorted(paths)

def load_and_resize_images(paths, image_size=(128,128)):
    imgs = []
    for p in paths:
        img = Image.open(p).convert('RGB')
        img = img.resize(image_size)
        imgs.append(np.array(img))
    return np.stack(imgs, axis=0)

def hog_features_from_images(X_images, pixels_per_cell=(16,16), cells_per_block=(2,2), orientations=9, resize=None):
    feats = []
    for img in X_images:
        if resize is not None:
            img = transform.resize(img, resize, anti_aliasing=True)
            img = (img * 255).astype('uint8')
        gray = color.rgb2gray(img)
        h = feature.hog(gray, orientations=orientations, pixels_per_cell=pixels_per_cell,
                        cells_per_block=cells_per_block, feature_vector=True)
        feats.append(h)
    return np.vstack(feats).astype(np.float32)

def save_features_to_disk(features, labels, prefix, out_dir='features'):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    x_path = out / f"{prefix}_X.npy"; y_path = out / f"{prefix}_y.npy"
    np.save(str(x_path), features); np.save(str(y_path), labels)
    return str(x_path), str(y_path)