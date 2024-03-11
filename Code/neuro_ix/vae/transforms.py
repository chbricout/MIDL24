from typing import Tuple
import numpy as np
import torch
from monai.transforms import (
    Transform
)
from torch.fft import (fftshift, fftn, ifftn)

class SelectMiddleAndSurroundingSlices(Transform):
    def __init__(self, spatial_dim=1):
        self.spatial_dim = spatial_dim

    def __call__(self, img_array:torch.Tensor):
        middle_index = img_array.shape[self.spatial_dim] // 2

        selected_slices = img_array[:,middle_index - 16 : middle_index + 16, :, :]
        return selected_slices
    
def create_disc_mask(shape, radius):
    rows, cols, depth = shape
    center = (rows // 2, cols // 2, depth//2)

    y, x, z = np.ogrid[:rows, :cols, :depth]
    distance = np.sqrt((x - center[1])**2 + (y - center[0])**2 + (z-center[2])**2)

    mask = distance <= radius
    return mask

class LowPassFilter(Transform):
    def __init__(self, radius:int, imsize:Tuple[int,int,int]):
        self.radius=radius
        self.mask = torch.Tensor(create_disc_mask(imsize, radius))

    def __call__(self, img_array:torch.Tensor):
        brain_FFT = fftn(img_array[0,:,:,:])
        brain_FFT_center = fftshift(brain_FFT)*self.mask # Apply masking to keep center of the FFT
        reconstruction = torch.abs(ifftn(brain_FFT_center))
        return torch.Tensor(reconstruction)[None,:,:,:]
    
class HighPassFilter(Transform):
    def __init__(self, radius:int, imsize:Tuple[int,int,int]):
        self.radius=radius
        self.mask = 1- torch.Tensor(create_disc_mask(imsize, radius))

    def __call__(self, img_array:torch.Tensor):
        brain_FFT = fftn(img_array[0,:,:,:])
        brain_FFT_edges = fftshift(brain_FFT)*self.mask # Apply masking to keep center of the FFT
        reconstruction = torch.abs(ifftn(brain_FFT_edges))
        return torch.Tensor(reconstruction)[None,:,:,:]
    
class HighFreqMask(Transform):
    def __init__(self, imsize:Tuple[int,int,int], radius=10, threshold=0.05):
        self.lowpass=LowPassFilter(radius, imsize)
        self.threshold = threshold
    def __call__(self, img_array:torch.Tensor):
        recon = self.lowpass(img_array)
        mask = torch.abs(recon-img_array) > 0.05
        return mask