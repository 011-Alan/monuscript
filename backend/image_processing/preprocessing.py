"""
MonuScript - Image Preprocessing Module
Adheres strictly to the Image Processing syllabus:
- EXIF orientation correction
- Grayscale conversion: I(x, y) = 0.299*R + 0.587*G + 0.114*B
- Standardized resizing and color space conversion (RGB, Grayscale, HSV)
"""

import io
import numpy as np
from PIL import Image, ImageOps
import cv2

# Standard dimensions for feature extraction pipeline
STANDARD_WIDTH = 256
STANDARD_HEIGHT = 256

def open_and_normalize_image(image_input):
    """
    Decodes an image from file path, file-like object, or bytes.
    Applies EXIF orientation correction and strips alpha channel cleanly.
    Returns:
        rgb_img: PIL.Image in RGB format
    """
    if isinstance(image_input, (str, bytes, bytearray, io.BytesIO)):
        if isinstance(image_input, (bytes, bytearray)):
            img = Image.open(io.BytesIO(image_input))
        else:
            img = Image.open(image_input)
    elif isinstance(image_input, Image.Image):
        img = image_input
    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")

    # Apply EXIF orientation if present
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass

    # Ensure RGB representation
    if img.mode == "RGBA":
        # Composite over a neutral white/light gray background to preserve edges
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        img = bg
    elif img.mode != "RGB":
        img = img.convert("RGB")

    return img

def rgb_to_grayscale_syllabus(rgb_array):
    """
    Converts RGB image to Grayscale using the exact course syllabus formula:
    I(x, y) = 0.299 * R(x, y) + 0.587 * G(x, y) + 0.114 * B(x, y)

    Parameters:
        rgb_array: np.ndarray of shape (H, W, 3) with values in [0, 255]
    Returns:
        gray_array: np.ndarray of shape (H, W) of uint8 [0, 255]
    """
    if rgb_array.ndim != 3 or rgb_array.shape[2] != 3:
        raise ValueError(f"Expected 3-channel RGB array, got shape {rgb_array.shape}")

    # Explicit floating-point weighted sum according to ITU-R BT.601 / syllabus
    R = rgb_array[:, :, 0].astype(np.float32)
    G = rgb_array[:, :, 1].astype(np.float32)
    B = rgb_array[:, :, 2].astype(np.float32)

    gray = 0.299 * R + 0.587 * G + 0.114 * B
    return np.clip(np.round(gray), 0, 255).astype(np.uint8)

def rgb_to_hsv(rgb_array):
    """
    Converts RGB array to HSV color space using OpenCV.
    Returns:
        hsv_array: np.ndarray of shape (H, W, 3) where:
            H in [0, 179]
            S in [0, 255]
            V in [0, 255]
    """
    if rgb_array.ndim != 3 or rgb_array.shape[2] != 3:
        raise ValueError("Expected 3-channel RGB array")
    return cv2.cvtColor(rgb_array, cv2.COLOR_RGB2HSV)

def preprocess_image_for_features(image_input, target_size=(STANDARD_WIDTH, STANDARD_HEIGHT)):
    """
    Full standard preprocessing pipeline for feature extraction:
    1. Open and normalize EXIF + RGBA -> RGB
    2. Resize to target size (using bilinear/bicubic interpolation)
    3. Convert to syllabus Grayscale
    4. Convert to HSV
    Returns dict:
        {
            "rgb": np.ndarray (H, W, 3) uint8,
            "gray": np.ndarray (H, W) uint8,
            "hsv": np.ndarray (H, W, 3) uint8,
            "original_pil": Image.Image,
            "original_size": (width, height)
        }
    """
    pil_img = open_and_normalize_image(image_input)
    orig_w, orig_h = pil_img.size

    # Resize to standard dimensions for uniform spatial and histogram feature extraction
    resized_pil = pil_img.resize(target_size, Image.Resampling.BILINEAR)
    rgb_arr = np.array(resized_pil, dtype=np.uint8)

    gray_arr = rgb_to_grayscale_syllabus(rgb_arr)
    hsv_arr = rgb_to_hsv(rgb_arr)

    return {
        "rgb": rgb_arr,
        "gray": gray_arr,
        "hsv": hsv_arr,
        "original_pil": pil_img,
        "original_size": (orig_w, orig_h)
    }
