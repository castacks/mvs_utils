
import numpy as np
import torch

_LOCAL_2_PI = 2 * np.pi

def check_valid_range(r0, r1, raise_exception=False):
    '''
    Return True if:
    - r0 and r1 are in the range of [-2pi, 2pi]
    - r0 < r1
    - r1 - r0 <= 2pi
    '''
    
    global _LOCAL_2_PI
    
    flag = True
    
    if r0 >= r1:
        flag = False
        reason = f'r0 ({r0}) >= r1 ({r1}). '
    elif r0 < -_LOCAL_2_PI or r1 > _LOCAL_2_PI:
        flag = False
        reason = f'r0 ({r0}) or r1 ({r1}) has a wrong value. The value must be in [-2pi, 2pi]. '
    elif r1 - r0 > _LOCAL_2_PI:
        flag = False
        reason = f'r1 ({r1}) - r0 ({r0}) = {r1-r0} > 2pi. '
    
    if not flag and raise_exception:
        raise ValueError(reason)
    
    return flag

def map_2_symmetric_range(angles):
    '''
    Put all the values of angels into the range of [-pi, pi]
    '''
    
    global _LOCAL_2_PI
    return ( angles + np.pi ) % _LOCAL_2_PI - np.pi

def shift_according_2_range(r0, r1, angles):
    '''
    This function returns a shifted version of angels such that the individual angles can be used
    to test if they fall in to the range defined by r0 and r1.
    
    There are some assumptions:
    - r0 and r1 are in the range of [-2pi, 2pi]
    - r0 < r1
    - r1 - r0 <= 2pi
    - The values of angles are in the range of [-pi, pi]
    
    angles: an NumPy array or PyTorch tensor. 
    '''
    
    global _LOCAL_2_PI
    if isinstance(angles, np.ndarray):
        return angles + _LOCAL_2_PI * ( 
                ( angles < r0 ).astype(angles.dtype) - ( angles > r1 ).astype(angles.dtype) )
    elif isinstance(angles, torch.Tensor):
        return angles + _LOCAL_2_PI * ( 
                ( angles < r0 ).to(dtype=angles.dtype) - ( angles > r1 ).to(dtype=angles.dtype) )
    else:
        raise TypeError(
            f'angles must be a NumPy array or a PyTorch tensor. '
            f'type(angles) = {type(angles)}. )')

def check_in_range(r0, r1, angles):
    '''
    Return a boolean array of the same shape as angles, indicating if each angle falls in the range.
    '''
    
    assert check_valid_range(r0, r1), f'Wrong range: [{r0}, {r1}]. '
    
    # Put all the values of angels into the range of [-pi, pi]
    angles = map_2_symmetric_range(angles)
    
    return ( angles >= r0 ) & ( angles <= r1 )
