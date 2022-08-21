
import copy
from numpy import poly
import torch
import math
import sys

from .shape_struct import ShapeStruct

CAMERA_MODELS = dict()

LOCAL_PI = math.pi

def register(dst):
    '''Register a class to a dstination dictionary. '''
    def dec_register(cls):
        dst[cls.__name__] = cls
        return cls
    return dec_register

def make_object(typeD, argD):
    '''Make an object from type collection typeD. '''

    assert( isinstance(typeD, dict) ), f'typeD must be dict. typeD is {type(typeD)}'
    assert( isinstance(argD,  dict) ), f'argD must be dict. argD is {type(argD)}'
    
    # Make a deep copy of the input dict.
    d = copy.deepcopy(argD)

    # Get the type.
    typeName = typeD[ d['type'] ]

    # Remove the type string from the input dictionary.
    d.pop('type')

    # Create the model.
    return typeName( **d )

def x2y2z_2_z_angle(x2, y2, z):
    '''
    Compute the angle (in radian) with respect to the z-axis.
    
    Arguments:
    x2 (Tensor or scalar): x**2.
    y2 (Tensor or scalar): y**2.
    z (Tensor or scalar): z.
    '''

    return torch.atan2( torch.sqrt( x2 + y2 ), z )

def xyz_2_z_angle(x, y, z):
    '''
    Compute the angle (in radian) with respect to the z-axis.

    Arguments:
    x (Tensor or scalar): x.
    y (Tensor or scalar): y.
    z (Tensor or scalar): z. 
    '''

    return x2y2z_2_z_angle(x**2.0, y**2.0, z)


class CameraModel(object):
    def __init__(self, name, fx, fy, cx, cy, fov_degree, shape_struct, in_to_tensor=False, out_to_numpy=False):
        super(CameraModel, self).__init__()

        self.name = name
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.fov_degree = fov_degree 
        self.fov_rad = self.fov_degree / 180.0 * LOCAL_PI
        
        if isinstance( shape_struct, dict ):
            self.ss = ShapeStruct( **shape_struct )
        elif isinstance( shape_struct, ShapeStruct ):
            self.ss = shape_struct
        else:
            raise Exception(f'shape_struct must be a dict or ShapeStruct object. Get {type(shape_struct)}')
        
        self.device = None
        self.in_to_tensor = in_to_tensor
        self.out_to_numpy = out_to_numpy

    @property
    def shape(self):
        return self.ss.shape

    def in_wrap(self, x):
        if self.in_to_tensor:
            return torch.as_tensor(x).to(device=self.device)
        else:
            return x

    def out_wrap(self, x):
        if self.out_to_numpy:
            return x.cpu().numpy()
        else:
            return x

    def pixel_2_ray(self, pixel_coor):
        '''
        Arguments:
        pixel_coor (Tensor): A 2xN Tensor contains the pixel coordinates. 
        
        NOTE: pixel_coor can also have a dimension of Bx2xN, where B is the 
        batch number.
        
        Returns:
        A 3xN Tensor representing the 3D rays. Bx3xN if batched.
        A (N,) Tensor representing the valid mask. BxN if batched.
        '''
        raise NotImplementedError()

    def point_3d_2_pixel(self, point_3d, normalized=False):
        '''
        Arguments:
        point_3d (Tensor): A 3xN Tensor contains 3D point coordinates. 
        normalized (bool): If True, then the returned coordinates are normalized to [-1, 1]
        
        NOTE: point_3d can also have a dimension of Bx3xN, where B is the 
        batch number.
        
        Returns: 
        A 2xN Tensor representing the 2D pixels. Bx2xN if batched.
        A (N,) Tensor representing the valid mask. BxN if batched.
        '''
        raise NotImplementedError()

    def to_(self, dtype=None, device=None):
        assert dtype is not None and device is not None, \
            f'dtype and device cannot both be None. '
        
        self.device = device

# Usenko, Vladyslav, Nikolaus Demmel, and Daniel Cremers. "The double sphere camera model." In 2018 International Conference on 3D Vision (3DV), pp. 552-560. IEEE, 2018.
@register(CAMERA_MODELS)
class DoubleSphere(CameraModel):
    def __init__(self, xi, alpha, fx, fy, cx, cy, fov_degree, shape_struct, in_to_tensor=False, out_to_numpy=False):
        super(DoubleSphere, self).__init__(
            'double_sphere', fx, fy, cx, cy, fov_degree, shape_struct, in_to_tensor=in_to_tensor, out_to_numpy=out_to_numpy)

        self.alpha = alpha
        self.xi = xi

        # w1 and w2 are defined in the origial paper.
        w1, w2 = self.get_w1_w2()
        self.w1 = w1
        self.w2 = w2

        self.r2_threshold = 1 / ( 2 * self.alpha - 1 )

    def get_w1_w2(self):
        # Refer to the original paper for more information.
        w1 = self.alpha / ( 1 - self.alpha ) \
            if self.alpha <= 0.5 \
            else ( 1 - self.alpha ) / self.alpha
        
        w2 = ( w1 + self.xi ) / math.sqrt( 2 * w1 * self.xi + self.xi**2.0 + 1 )

        return w1, w2

    def to_(self, dtype=None, device=None):
        super().to_(dtype, device)
        # Do nothing.

    def pixel_2_ray(self, pixel_coor):
        '''
        Arguments:
        pixel_coor (Tensor): A 2xN Tensor contains the pixel coordinates.
        
        NOTE: pixel_coor can also have a dimension of Bx2xN, where B is the 
        batch number. 
        
        Returns:
        ray: A 3xN Tensor representing the 3D rays. Bx3xN if batched.
        valid_mask: A (N,) Tensor representing the valid mask. BxN if batched.
        '''
        
        pixel_coor = self.in_wrap(pixel_coor)
        
        # mx and my becomes float64 if pixel_coor.dtype is integer type.
        mx = ( pixel_coor[..., 0, :] - self.cx ) / self.fx
        my = ( pixel_coor[..., 1, :] - self.cy ) / self.fy
        r2 = mx**2.0 + my**2.0

        if ( self.alpha <= 0.5 ):
            valid_mask = torch.full((mx.size,), True)
        else:
            valid_mask = r2 <= self.r2_threshold

        # Suppress the waring from the invalid values.
        r2[torch.logical_not(valid_mask)] = 0

        mz = \
            ( 1 - self.alpha**2.0 * r2 ) / \
            ( self.alpha * torch.sqrt( 1 - ( 2*self.alpha - 1 ) * r2 ) + 1 - self.alpha )

        mz2 = mz**2.0

        t = ( mz * self.xi + torch.sqrt( mz2 + ( 1 - self.xi**2.0 ) * r2 ) ) / ( mz2 + r2 )
        x = t * mx
        y = t * my
        z = t * mz - self.xi

        # Need to deal with batch dim
        ray = torch.stack( (x, y, z), dim=-2 )

        # Compute the norm of ray along column direction.
        # norm_ray = torch.linalg.norm( ray, ord=2, dim=0, keepdim=True ) # Non-batched version
        norm_ray = torch.linalg.norm( ray, ord=2, dim=-2, keepdim=True )
        zero_mask = norm_ray == 0
        norm_ray[zero_mask] = 1

        # Normalize ray.
        ray = ray / norm_ray

        # Filter by FOV.
        a = xyz_2_z_angle( x, y, z )
        valid_mask = torch.logical_and(
            valid_mask, 
            a <= self.fov_rad / 2.0
        )

        return self.out_wrap(ray), self.out_wrap(valid_mask)

    def point_3d_2_pixel(self, point_3d, normalized=False):
        '''
        Arguments:
        point_3d (Tensor): A 3xN Tensor contains 3D point coordinates. 
        normalized (bool): If True, then the returned coordinates are normalized to [-1, 1]
        
        NOTE: point_3d can also have a dimension of Bx3xN, where B is the 
        batch number. 
        
        Returns: 
        pixel_coor: A 2xN Tensor representing the 2D pixels. Bx2xN if batched.
        valid_mask: A (N,) Tensor representing the valid mask. BXN if batched.
        '''

        point_3d = self.in_wrap(point_3d)

        # torch.split results in Bx1XN.
        x, y, z = torch.split( point_3d, 1, dim=-2 )

        x2 = x**2.0 # Note: this may promote x2 to torch.float64 if point_3d.dtype=torch.int. 
        y2 = y**2.0
        z2 = z**2.0

        d1 = torch.sqrt( x2 + y2 + z2 )
        d2 = torch.sqrt( x2 + y2 + ( self.xi * d1 + z )**2.0 )

        # Pixel coordinates. 
        t = self.alpha * d2 + ( 1 - self.alpha ) * ( self.xi * d1 + z )
        px = self.fx / t * x + self.cx
        py = self.fy / t * y + self.cy
        if normalized:
            px = px / ( self.ss.W - 1 ) * 2 - 1
            py = py / ( self.ss.H - 1 ) * 2 - 1

        # pixel_coor = torch.stack( (px, py), dim=0 )
        pixel_coor = torch.cat( (px, py), dim=-2 )

        # Filter the invalid pixels.
        valid_mask = z > -self.w2 * d1

        # Filter by FOV.
        a = x2y2z_2_z_angle( x2, y2, z )
        valid_mask = torch.logical_and(
            valid_mask, 
            a <= self.fov_rad / 2.0
        )
        
        # This is for the batched dimension.
        valid_mask = valid_mask.squeeze(-2)

        return self.out_wrap(pixel_coor), self.out_wrap(valid_mask)

@register(CAMERA_MODELS)
class Equirectangular(CameraModel):
    def __init__(self, cx, cy, shape_struct, lon_shift=0, open_span=False, in_to_tensor=False, out_to_numpy=False):
        super(Equirectangular, self).__init__(
            'equirectangular', 1, 1, cx, cy, 360, shape_struct, in_to_tensor=in_to_tensor, out_to_numpy=out_to_numpy)

        self.lon_shift = lon_shift
        self.longitude_span = torch.Tensor( [ -LOCAL_PI,   LOCAL_PI ]  ).to(dtype=torch.float32) + self.lon_shift
        self.latitude_span  = torch.Tensor( [ -LOCAL_PI/2, LOCAL_PI/2] ).to(dtype=torch.float32)
        
        # Since lon_shift is applied by adding to the longitude span, the shifted frame has a measured
        # rotation of -lon_shift, w.r.t. the original frame. Thus, the shifted frame has a rotation 
        # that is measured in the original frame as
        a = -self.lon_shift
        self.R_ori_shifted = torch.Tensor(
            [ [ math.cos(a), -math.sin(a) ], 
              [ math.sin(a),  math.cos(a) ] ]
            ).to(dtype=torch.float32)

        # open_span is True means the last column of pixels do not have the same longitude angle as the first column.
        self.open_span = open_span
        
        # The actual longitude span that all the pixels cover.
        self.lon_span_pixel = ( self.longitude_span[1] - self.longitude_span[0] )
        if self.open_span:
            self.lon_span_pixel = 2*self.cx / ( 2*self.cx + 1 ) * self.lon_span_pixel

    def to_(self, dtype=None, device=None):
        super().to_(dtype, device)
        
        self.longtitude_span = self.longitude_span.to(dtype, device)
        self.latitude_span   = self.latitude_span.to(dtype, device)
        self.R_ori_shifted   = self.R_ori_shifted.to(dtype, device)

    def pixel_2_ray(self, pixel_coor):
        '''
        Assuming cx and cy is the center coordinates of the image. 
        Thus, the image shape is [ 2*cy + 1, 2*cx + 1 ]
        
        Arguments:
        pixel_coor (Tensor): A 2xN Tensor contains the pixel coordinates. 
        
        NOTE: pixel_coor can also have a dimension of Bx2xN, where B is the 
        batch number. 
        
        Returns:
        ray: A 3xN Tensor representing the 3D rays. Bx3XN if batched.
        valid_mask: A (N,) Tensor representing the valid mask. BxN if batched.
        '''
        
        pixel_coor = self.in_wrap(pixel_coor)
        
        pixel_space_center = \
            torch.Tensor([ self.cx, self.cy ]).to(dtype=torch.float32, device=pixel_coor.device).view((2, 1))
        angle_start = \
            torch.Tensor([ self.longitude_span[0], self.latitude_span[0] ]).to(dtype=torch.float32, device=pixel_coor.device).view((2, 1))
        
        angle_span = torch.Tensor(
            [ self.lon_span_pixel, self.latitude_span[1] - self.latitude_span[0] ]
        ).to(dtype=torch.float32, device=pixel_coor.device).view((2, 1))
        
        # lon_lat.dtype becomes torch.float64 if pixel_coor.dtype=torch.int.
        lon_lat = pixel_coor / ( 2 * pixel_space_center ) * angle_span + angle_start
        
        # Bx1xN after calling torch.split.
        longitute, latitute = torch.split( lon_lat, 1, dim=-2 )
        
        c = torch.cos(latitute)
        
        x = c * torch.sin(longitute)
        y =     torch.sin(latitute)
        z = c * torch.cos(longitute)
        
        # return self.out_wrap( torch.stack( (x, y, z), dim=0 )   ), \
        #        self.out_wrap( torch.ones_like(x).to(torch.bool) )
               
        return self.out_wrap( torch.cat( (x, y, z), dim=-2 )   ), \
               self.out_wrap( torch.ones_like(x.squeeze(-2)).to(torch.bool) )
    
    def point_3d_2_pixel(self, point_3d, normalized=False):
        '''
        Arguments:
        point_3d (Tensor): A 3xN Tensor contains 3D point coordinates. 
        normalized (bool): If True, then the returned coordinates are normalized to [-1, 1]
        
        NOTE: point_3d can also have a dimension of Bx3xN, where B is the 
        batch number. 
        
        Returns: 
        pixel_coor: A 2xN Tensor representing the 2D pixels. Bx2xN if batched.
        valid_mask: A (N,) Tensor representing the valid mask. BxN if batched.
        '''
        
        point_3d = self.in_wrap(point_3d)
        
        r = torch.linalg.norm(point_3d, dim=-2)
        
        lat = torch.asin(point_3d[..., 1, :] / r)
        
        z_x = self.R_ori_shifted @ point_3d[ ..., [2, 0], : ]
        lon = torch.atan2( z_x[..., 1, :], z_x[..., 0, :] )
        
        if normalized:
            p_y = lat / LOCAL_PI * 2
            p_x = ( lon + LOCAL_PI ) / self.lon_span_pixel * 2 - 1
        else:
            p_y = lat / LOCAL_PI + 0.5 # [ 0, 1 ]
            p_x = ( lon + LOCAL_PI ) / self.lon_span_pixel # [ 0, 1 ], closed span
            
            p_y = p_y * ( 2 * self.cy )
            p_x = p_x * ( 2 * self.cx )
        
        return self.out_to_numpy( torch.stack( (p_x, p_y), dim=-2 ) ), \
               self.out_to_numpy( torch.ones_like(p_x).to(torch.bool) )

@register(CAMERA_MODELS)
class Ocam(CameraModel):
    EPS = sys.float_info.epsilon
    
    def __init__(self, poly_coeff, inv_poly_coeff, cx, cy, affine_coeff, fov_degree, shape_struct, in_to_tensor=False, out_to_numpy=False):
        '''
        The implementation is mostly based on 
        https://github.com/hyu-cvlab/omnimvs-pytorch/blob/3016a5c01f55c27eff3c019be9aee02e34aaaade/utils/ocam.py#L15
        
        When reading values for poly_coeff and inv_poly_coeff, make sure that the coefficients of 
        higher order are listed first. If these values are read from the yaml file provided by omnimvs
        model, then we need to reverse the order of the original data (also skip the first value that is
        showing the total number of coefficients).
        
        The CIF (Camera Image Frame) defined by Davide Scaramuzza (see below) if different than ours. 
        The CIF here is originally defined as z-backward, y-right, and x-downward. So we need to convert
        between our CIF and the CIF used by Davide when dealing with coordinates.
        
        Note that we are not fliping the order of cx and cy internally, meaning the input arguments must have the 
        correct order (fliped outside this class) and we need to use self.cy for Davide's x-axis.
        
        The Ocam model is described by Davide Scaramuzza at
        https://sites.google.com/site/scarabotix/ocamcalib-omnidirectional-camera-calibration-toolbox-for-matlab
        '''
        
        super().__init__('Ocam', 1, 1, cx, cy, fov_degree, shape_struct, in_to_tensor=in_to_tensor, out_to_numpy=out_to_numpy)
        
        # Polynomial coefficients starting from the highest degree.
        self.poly_coeff     = torch.as_tensor(poly_coeff).to(dtype=torch.float32)     # Only contains the coefficients.
        self.inv_poly_coeff = torch.as_tensor(inv_poly_coeff).to(dtype=torch.float32) # Only contains the coefficients.
        self.affine_coeff   = affine_coeff   # c, d, e
        
    def to_(self, dtype=None, device=None):
        super().to_(dtype, device)
    
        self.poly_coeff     = self.poly_coeff.to(dtype, device)
        self.inv_poly_coeff = self.inv_poly_coeff.to(dtype, device)
    
    @staticmethod
    def poly_eval(poly_coeff, x):
        '''
        Evaluate the polynomial.
        '''
        # Exponent.
        p = torch.arange(len(poly_coeff)-1, -1, -1, device=x.device).view((-1, 1))
        
        # Change shapes.
        poly_coeff = poly_coeff.view((-1, 1))
        
        # Consider the batch dimension.
        # x = x.view((1, -1))
        # N -> 1xN, BxN -> Bx1xN
        x = x.unsqueeze(-2)
        
        # return torch.sum( poly_coeff * x ** p, dim=0 )
        return torch.sum( poly_coeff * x ** p, dim=-2 )
        
    def pixel_2_ray(self, pixel_coor):
        '''
        Arguments:
        pixel_coor (Tensor): A 2xN Tensor contains the pixel coordinates. 
        
        NOTE: pixel_coor can also have a dimension of Bx2xN, where B is the 
        batch number. 
        
        Returns:
        ray: A 3xN Tensor representing the 3D rays. Bx3xN if batched.
        valid_mask: A (N,) Tensor representing the valid mask. BxN if batched.
        '''

        pixel_coor = self.in_wrap(pixel_coor).to(dtype=torch.float32)

        p = torch.zeros_like(pixel_coor, device=pixel_coor.device)
        
        # We need to use Davide's definition of the coordinate system.
        p[..., 0, :] = pixel_coor[..., 1, :] - self.cy
        p[..., 1, :] = pixel_coor[..., 0, :] - self.cx
        
        c, d, e = self.affine_coeff
        invdet = 1.0 / (c - d * e)
        
        A_inv = invdet * torch.Tensor( [
            [  1, -d ], 
            [ -e,  c ] ] ).to(dtype=pixel_coor.dtype, device=pixel_coor.device)
        
        # A_inv = invdet * torch.Tensor( [
        #     [ -d,  1 ], 
        #     [  c, -e ] ] ).to(dtype=pixel_coor.dtype, device=pixel_coor.device)

        p = A_inv @ p
        
        x = p[..., 0, :]
        y = p[..., 1, :]
        
        rho = torch.sqrt( x**2 + y**2 )

        z = Ocam.poly_eval( self.poly_coeff, rho )
        
        # theta is angle from the optical axis.
        theta = torch.atan2(rho, -z)
        
        # Convert back to our coordinate system.
        # out   = torch.stack((x, y, -z), dim=0)
        # out   = torch.stack((y, x, -z), dim=0)
        out   = torch.stack((y, x, -z), dim=-2)
        
        max_theta = self.fov_rad / 2.0
        valid_mask = theta <= max_theta

        return self.out_wrap( out ), \
               self.out_wrap( valid_mask )
        
    def point_3d_2_pixel(self, point_3d, normalized=False):
        '''
        Arguments:
        point_3d (Tensor): A 3xN Tensor contains 3D point coordinates. 
        normalized (bool): If True, then the returned coordinates are normalized to [-1, 1]
        
        NOTE: point_3d can also have a dimension of Bx3xN, where B is the 
        batch number. 
        
        Returns: 
        pixel_coor: A 2xN Tensor representing the 2D pixels. Bx2xN if batched.
        valid_mask: A (N,) Tensor representing the valid mask. BxN if batched.
        '''   
        
        point_3d = self.in_wrap(point_3d)
        
        # torch.split() will reserve the dimension.
        x_3d = point_3d[..., 0, :]
        y_3d = point_3d[..., 1, :]
        z_3d = point_3d[..., 2, :]
        
        norm  = torch.sqrt( x_3d**2 + y_3d**2 ) + Ocam.EPS
        theta = torch.atan2( -z_3d, norm )
        rho   = Ocam.poly_eval( self.inv_poly_coeff, theta )
        
        # max_theta check : theta is the angle from xy-plane in ocam, 
        # thus add pi/2 to compute the angle from the optical axis.
        theta = theta + LOCAL_PI / 2
        
        c, d, e = self.affine_coeff
        
        # We need to use Davide's definition of the coordinate system.
        y = x_3d / norm * rho
        x = y_3d / norm * rho
        x2 = x * c + y * d + self.cy
        y2 = x * e + y     + self.cx
        
        # Convert back to our coordinate system.
        if normalized:
            y2 = y2 / ( self.ss.W - 1 ) * 2 - 1
            x2 = x2 / ( self.ss.H - 1 ) * 2 - 1
        
        out = torch.stack( (y2, x2), dim=-2 )
        
        return self.out_to_numpy( out ), \
               self.out_to_numpy( theta <= self.fov_rad / 2.0 )
    