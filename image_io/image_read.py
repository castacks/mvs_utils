import cv2
import numpy as np

from .float_type import c4_uint8_as_float

SIX_IMG_SIDES = [ 'front', 'right', 'bot', 'left', 'top', 'back' ]
SIX_IMG_NAMES = [ 'front', 'right', 'bottom', 'left', 'top', 'back' ]

def read_image(fn, dtype=np.uint8):
    img = cv2.imread(fn, cv2.IMREAD_UNCHANGED)
    assert img is not None, f'Failed to read {fn}. '
    return img.astype(dtype)

def read_image_gray(fn):
    img = cv2.imread(fn, cv2.IMREAD_GRAYSCALE)
    assert img is not None, f'Failed to read {fn}. '
    return img

def read_mask(fn, valid_value=255):
    return read_image_gray(fn) == valid_value

def read_compressed_float(fn):
    img = read_image(fn, np.uint8)
    return np.squeeze( 
        c4_uint8_as_float(img), 
        axis=-1 )

def read_six_image(fn):
    fn_splt = fn.split(".")
    file_type = fn_splt[1]
    base_path = fn_splt[0]

    return np.array([read_image(base_path + "_" + s + "." + file_type) for s in SIX_IMG_SIDES])

def read_compressed_float_siximg(fn):
    fn_splt = fn.split(".")
    file_type = fn_splt[1]
    base_path = fn_splt[0]

    return np.array([read_compressed_float(base_path + "_" + s + "." + file_type) for s in SIX_IMG_SIDES])

def siximg_arr_to_imgdict(arr):
    splt = list(arr)
    out = dict()

    for i,s in enumerate(SIX_IMG_NAMES):
        out.update({s:splt[i]})

    return out