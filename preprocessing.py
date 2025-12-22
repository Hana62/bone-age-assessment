import cv2
import numpy as np

def read_xray_grayscale(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img

def optical_normalization(img: np.ndarray) -> np.ndarray:
    # Contrast enhancement using CLAHE (robust histogram equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img)

def denoise(img: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(img, (5, 5), 0)

def hand_mask(img: np.ndarray) -> np.ndarray:
    # Otsu threshold to separate hand from background
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Ensure hand is white (foreground)
    if np.mean(th) > 127:
        th = 255 - th

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
    return th

def largest_component_bbox(mask: np.ndarray):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    return x, y, w, h

def crop_hand_roi(img: np.ndarray) -> np.ndarray:
    m = hand_mask(img)
    bb = largest_component_bbox(m)
    if bb is None:
        return img
    x, y, w, h = bb

    # Add a small padding
    pad = int(0.03 * max(w, h))
    x0 = max(0, x - pad); y0 = max(0, y - pad)
    x1 = min(img.shape[1], x + w + pad); y1 = min(img.shape[0], y + h + pad)
    return img[y0:y1, x0:x1]

def geometric_normalization(img: np.ndarray, out_size=(512, 512)) -> np.ndarray:
    # Basic geometric normalization: resize to fixed size
    return cv2.resize(img, out_size, interpolation=cv2.INTER_AREA)

def preprocess_image(path: str, out_size=(512, 512)) -> np.ndarray:
    img = read_xray_grayscale(path)
    img = denoise(img)
    img = optical_normalization(img)
    img = crop_hand_roi(img)
    img = geometric_normalization(img, out_size=out_size)
    return img

def extract_rois(preprocessed_img: np.ndarray):
    '''
    Lightweight ROI strategy aligned with your documents:
    - Whole hand
    - Wrist region (lower part)
    - Finger/epiphysis region (upper-middle)

    This is a practical approximation without manual landmarks.
    '''
    img = preprocessed_img
    h, w = img.shape

    whole = img

    # Wrist: bottom 35%
    wrist = img[int(h * 0.65):h, :]

    # Fingers/epiphysis: top 45%, central 70%
    y0, y1 = 0, int(h * 0.45)
    x0, x1 = int(w * 0.15), int(w * 0.85)
    fingers = img[y0:y1, x0:x1]

    # Normalize ROI sizes (consistent feature vectors)
    wrist = cv2.resize(wrist, (256, 256), interpolation=cv2.INTER_AREA)
    fingers = cv2.resize(fingers, (256, 256), interpolation=cv2.INTER_AREA)
    whole = cv2.resize(whole, (256, 256), interpolation=cv2.INTER_AREA)

    return {
        "whole": whole,
        "wrist": wrist,
        "fingers": fingers,
    }
