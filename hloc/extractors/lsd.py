import copy
import numpy as np
import torch
import pytlsd
import pytlbd
from ..utils.base_model import BaseModel
import cv2

EPS = 1e-6


def process_pyramid(img, detector, n_levels=5, level_scale=np.sqrt(2), presmooth=True):
    octave_img = img.copy()
    pre_sigma2 = 0
    cur_sigma2 = 1.0
    pyramid = []
    multiscale_segs = []
    for i in range(n_levels):
        increase_sigma = np.sqrt(cur_sigma2 - pre_sigma2)
        blurred = cv2.GaussianBlur(octave_img, (5, 5), increase_sigma, borderType=cv2.BORDER_REPLICATE)
        pyramid.append(blurred)

        if presmooth:
            multiscale_segs.append(detector(blurred))
        else:
            multiscale_segs.append(detector(octave_img))
        
        # down sample the current octave image to get the next octave image
        new_size = (int(octave_img.shape[1] / level_scale), int(octave_img.shape[0] / level_scale))
        octave_img = cv2.resize(blurred, new_size, 0, 0, interpolation=cv2.INTER_NEAREST)
        pre_sigma2 = cur_sigma2
        cur_sigma2 = cur_sigma2 * 2

    return multiscale_segs, pyramid


class LSD(BaseModel):
    default_conf = {
        'n_levels': 5,
        'level_scale': np.sqrt(2),
        'presmooth': False
    }
    required_inputs = ['image']

    def _init(self, conf):
        self.n_levels = conf['n_levels']
        self.level_scale = conf['level_scale']
        self.presmooth = conf['presmooth']

    def _forward(self, data):
        image = data['image'].cpu().numpy()
        assert image.shape[1] == 1
        assert image.min() >= -EPS and image.max() <= 1 + EPS
    
        img8 = (image[0,0] * 255).astype(np.uint8)
        multiscale_segs, pyramid = process_pyramid(img8, pytlsd.lsd, n_levels=self.n_levels, level_scale=self.level_scale, presmooth=self.presmooth)
        multiscale = pytlbd.merge_multiscale_segs(multiscale_segs)
        descriptors_ms = pytlbd.lbd_multiscale_pyr(pyramid, multiscale, 9, 7)
        line_segs = np.array([msl[0][1] * self.level_scale ** msl[0][0] for msl in multiscale])

        if len(multiscale) == 0:
            scores = np.array([])
            line_segs = np.zeros((0,4))
            descriptors = np.zeros((72,0))
        else:
            scores = line_segs[:,4]
            #num_pixels = line_segs[:,5]
            line_segs = line_segs[:,0:4] # x1 y1 x2 y2        
            
            # TODO: Figure out how to handle store multi-scale descriptors
            #       For now we only take one descriptor for each segment
            descriptors = np.array([x[0] for x in descriptors_ms]).transpose()
                
        return {            
            'line_segments': torch.from_numpy(line_segs)[None],            
            'scores': torch.from_numpy(scores)[None],
            'descriptors':  torch.from_numpy(descriptors)[None],
        }
