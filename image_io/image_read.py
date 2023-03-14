import cv2
import numpy as np

from .float_type import c4_uint8_as_float

SIX_IMG_SIDES = [ 'front', 'right', 'bot', 'left', 'top', 'back' ]
SIX_IMG_NAMES = [ 'front', 'right', 'bottom', 'left', 'top', 'back' ]

def enforce_3_channel(img):
    '''
    Assuming img is an NumPy array.
    '''
    assert img.ndim in (2, 3), f'Image must be 2D or 3D. Got img.shape = {img.shape}. '
    return np.expand_dims(img, axis=-1) if img.ndim == 2 else img

def read_image(fn, dtype=np.uint8, enforce_3_c=False):
    img = cv2.imread(fn, cv2.IMREAD_UNCHANGED)
    assert img is not None, f'Failed to read {fn}. '
    img = img.astype(dtype)
    return enforce_3_channel(img) if enforce_3_c else img

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

    return np.array( [ 
            read_image(base_path + "_" + s + "." + file_type, enforce_3_c=True)
            for s in side_suffixes ] )

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

class GroupedImageShape(object):
    '''
    This calss is for representing the shape of a group of images and reduce the difficulty of
    writing if-statements for checking if there is a group dimension or channel dimension.
    
    GroupedImageShape always has the group dimension and the channel dimensino.
    
    If the user knows that the image is a grayscale image and explicitly requires a 2-element
    list as the shape object, then use the shape_stripped property. When using shape_stripped 
    on a non-grayscale image shape, an assertion error will be raised.
    '''
    
    def __init__(self, img_shape, group_size=1):
        n = len(img_shape)
        assert n in (2, 3), f'Image shape must be 2D or 3D. Got img_shape = {img_shape}. '
        self.img_shape = img_shape if n == 3 else [ *img_shape, 1 ]
        self.group_size = group_size
        self.grouped_shape = [ self.group_size, *self.img_shape ]
        
    @property
    def shape(self):
        '''
        Guranteed to be 3-element.
        '''
        return self.img_shape
    
    @property
    def shape_stripped(self):
        assert len(self.img_shape) == 3 and self.img_shape[-1] == 1, \
            f'Cannot strip. self.img_shape = {self.img_shape}. '
        return self.img_shape[:-1]

    def __getitem__(self, key):
        return self.grouped_shape[key]
    
    def __len__(self):
        return len(self.grouped_shape)
    