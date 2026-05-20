import cv2
import numpy as np
from PIL import Image

_MIN_WIDTH = 1000


def preprocess(image: Image.Image) -> Image.Image:
    """Prepare a PIL image for PaddleOCR: upscale, denoise, binarise."""
    img = np.array(image.convert("L"))

    h, w = img.shape
    if w < _MIN_WIDTH:
        scale = _MIN_WIDTH / w
        img = cv2.resize(
            img,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_CUBIC,
        )

    img = cv2.fastNlMeansDenoising(img, h=10)

    img = cv2.adaptiveThreshold(
        img, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=31,
        C=11,
    )

    return Image.fromarray(img).convert("RGB")
