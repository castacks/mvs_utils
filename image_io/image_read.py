import cv2
import numpy as np

from .float_type import c4_uint8_as_float

def read_image(fn, dtype=np.uint8):
    img = cv2.imread(fn, cv2.IMREAD_UNCHANGED)
    assert img is not None, f'Failed to read {fn}'
    return img.astype(dtype)

def read_compressed_float(fn):
    img = read_image(fn, np.uint8)
    return np.squeeze( 
        c4_uint8_as_float(img), 
        axis=-1 )