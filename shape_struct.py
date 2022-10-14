
import numpy as np

class ShapeStruct(object):
    def __init__(self, H, W):
        super().__init__()
        
        self.H = H
        self.W = W
        
    @property
    def shape(self):
        '''
        This funtion is meant to be used with NumPy, PyTorch, etc.
        '''
        return (self.H, self.W)
    
    @property
    def size(self):
        '''
        This function is meant to be used with OpenCV APIs.
        '''
        return (self.W, self.H)
    
    @property
    def shape_numpy(self):
        return np.array( [ self.H, self.W ], dtype=np.int32 )
    
    @staticmethod
    def read_shape_struct(dict_like):
        '''
        Read shape information from a dict-like object.
        '''
        return ShapeStruct( H=dict_like['H'], W=dict_like['W'] ) \
            if not isinstance(dict_like, ShapeStruct) \
            else dict_like

    def __repr__(self) -> str:
        return f'ShapeStruct(H={self.H}, W={self.W})'