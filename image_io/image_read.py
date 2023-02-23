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

def read_six_image(fn_template, side_suffixes=SIX_IMG_SIDES):
    fn_splt = fn_template.split(".")
    file_type = fn_splt[1]
    base_path = fn_splt[0]

    return np.array([read_image(base_path + "_" + s + "." + file_type) for s in side_suffixes])

def read_compressed_float_siximg(fn_template, side_suffixes=SIX_IMG_SIDES):
    fn_splt = fn_template.split(".")
    file_type = fn_splt[1]
    base_path = fn_splt[0]

    return np.array(
        [read_compressed_float(base_path + "_" + s + "." + file_type) for s in side_suffixes] )

def siximg_arr_to_imgdict(arr, side_names=SIX_IMG_NAMES):
    '''
    Note: This function assumes that the element order of arr is consistent with side_names.
    
    arr: NumPy Array of shape (6, H, W, C) or a list of length 6 with elements of shape (H, W, C).
    '''
    # # Enforce splt to be a list.
    # splt = list(arr)
    # out = dict()
    # for i, s in enumerate(side_names):
    #     out.update({s:splt[i]})

    return dict( zip(side_names, arr) )