import torch
import cv2

import numpy as np

class Depth2Distance(object):

    def __init__(self, camera_model):
        self.camera_model = camera_model
        self.pixel_coords = self.camera_model.pixel_coordinates()
        self.pixel_coords = self.pixel_coords.reshape((*self.camera_model.shape,2)).cuda()

    def __call__(self, z):

        z = torch.from_numpy(z).cuda().squeeze(2)
        xyz = torch.zeros((*self.pixel_coords.shape[:2],3))

        u = self.pixel_coords[:,:,0]
        v = self.pixel_coords[:,:,1]

        fx = self.camera_model.fx
        fy = self.camera_model.fy
        cx = self.camera_model.cx
        cy = self.camera_model.cy

        xyz[:,:,0] = (z / fx) * (u - cx)
        xyz[:,:,1] = (z / fy) * (v - cy)
        xyz[:,:,2] = z

        distance = torch.norm(xyz,dim=2)

        return distance.cpu().numpy()