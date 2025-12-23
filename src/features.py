import numpy as np
import cv2
from skimage.feature import hog

def intensity_stats(img: np.ndarray) -> np.ndarray:
    imgf = img.astype(np.float32) / 255.0
    return np.array([
        imgf.mean(),
        imgf.std(),
        np.quantile(imgf, 0.10),
        np.quantile(imgf, 0.50),
        np.quantile(imgf, 0.90),
    ], dtype=np.float32)

def histogram_features(img: np.ndarray, bins: int = 32) -> np.ndarray:
    hist = cv2.calcHist([img], [0], None, [bins], [0, 256]).reshape(-1)
    hist = hist.astype(np.float32)
    hist /= (hist.sum() + 1e-6)
    return hist

def hog_features(img: np.ndarray) -> np.ndarray:
    # HOG captures shape/edge structure (useful for bone patterns)
    feats = hog(
        img,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    )
    return feats.astype(np.float32)

def extract_features(rois: dict) -> np.ndarray:
    '''
    Concatenate features from multiple ROIs (feature-level integration),
    then we will do decision-level fusion between models.
    '''
    all_feats = []
    for name in ["whole", "wrist", "fingers"]:
        img = rois[name]
        all_feats.append(intensity_stats(img))
        all_feats.append(histogram_features(img, bins=32))
        all_feats.append(hog_features(img))
    return np.concatenate(all_feats, axis=0)
