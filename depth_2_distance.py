import numpy as np
import torch

class Depth2Distance(object):

    def __init__(self, camera_model):
        self.camera_model = camera_model
        self._device = self.camera_model.device
        
        # Pixel coordiantes in [2, H, W]. Contiguous by default.
        self.pixel_coords = self.camera_model.pixel_coordinates()

        self.xyz = torch.zeros( 
                ( 3, *self.pixel_coords.shape[:2] ), device=self._device, dtype=torch.float32 )
        
    @property
    def device(self):
        return self._device
    
    @device.setter
    def device(self, d):
        self.camera_model.device = d
        self.pixel_coords = self.pixel_coords.to(d)
        self._device = d

    def __call__(self, z):
        '''
        z: NumPy array [H, W, 1] or PyTorch tensor [1, H, W]. 
        '''

        if isinstance(z, np.ndarray):
            z = torch.from_numpy(z).permute((2, 0, 1))
        
        if not isinstance(z, torch.Tensor):
            raise TypeError(f'z must be a NumPy array or PyTorch tensor, but got {type(z)}. ')
        
        z = z.to(device=self.device).squeeze(0)
        
        xyz = self.xyz.detach().clone()

        u = self.pixel_coords[0, :, :]
        v = self.pixel_coords[1, :, :]

        fx = self.camera_model.fx
        fy = self.camera_model.fy
        cx = self.camera_model.cx
        cy = self.camera_model.cy

        xyz[0, :, :] = (z / fx) * (u - cx)
        xyz[1, :, :] = (z / fy) * (v - cy)
        xyz[2, :, :] = z

        distance = torch.norm(xyz, dim=0)

        return distance.cpu().numpy()