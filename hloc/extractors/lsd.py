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
        'presmooth': False,
        'max_n_lines': 512
    }
    required_inputs = ['image']

    def _init(self, conf):
        self.n_levels = conf['n_levels']
        self.level_scale = conf['level_scale']
        self.presmooth = conf['presmooth']
        self.max_lines = conf['max_n_lines']

    @staticmethod
    def to_multiscale_lines(lines):
        ms_lines = []
        for l in lines.reshape(-1, 4):
            ll = np.append(l, [0, np.linalg.norm(l[:2] - l[2:4])])
            ms_lines.append([(0, ll)] + [(i, ll / (i * np.sqrt(2))) for i in range(1, 5)])
        return ms_lines

    def _forward(self, data):
        image = data['image'].cpu().numpy()
        assert image.shape[1] == 1
        assert image.min() >= -EPS and image.max() <= 1 + EPS

        img8 = (image[0, 0] * 255).astype(np.uint8)
        left_multiscale_segs, left_pyramid = process_pyramid(img8, pytlsd.lsd, n_levels=self.n_levels,
                                                             level_scale=self.level_scale, presmooth=self.presmooth)
        ms_lines = pytlbd.merge_multiscale_segs(left_multiscale_segs)

        if len(ms_lines) == 0:
            return {'lines': torch.zeros((1, 0, 4), dtype=torch.float),
                    'scores': torch.zeros((1, 0), dtype=torch.float),
                    'descriptors': torch.zeros((1, 0, 5, 72), dtype=torch.float)}

        lines = np.array([msl[0][1] * np.sqrt(2) ** msl[0][0] for msl in ms_lines])
        lengths = np.linalg.norm(lines[:, 2:4] - lines[:, 0:2], axis=1)

        n_scales = np.array([len(msl) for msl in ms_lines])
        # The LSD importance is the log(NFA), here we multiply it by the sqrt(2)**scale to compensate the scale
        importance = np.array([np.array([sl[1][-2] * np.sqrt(2) ** sl[0] for sl in msl]).mean() for msl in ms_lines])
        # Lets score segments taking into account how robust they are to scale changes (n_scales),
        # how many aligned pixels they have (importance) and their length
        scores = np.log(n_scales * lengths * importance)

        # Take the most relevant segments with
        indices = np.argsort(-scores)
        scores = scores[indices]
        lines = lines[indices, :4]
        if self.max_lines is not None:
            lines = lines[:self.max_lines]
            scores = scores[:self.max_lines]

        # We will describe always the same number of scales to make the descriptor easier to store
        ms_lines = self.to_multiscale_lines(lines)

        descriptors = pytlbd.lbd_multiscale_pyr(left_pyramid, ms_lines, 9, 7)
        return {
            'lines': torch.from_numpy(lines)[None],
            'scores': torch.from_numpy(scores)[None],
            'descriptors': torch.from_numpy(np.array(descriptors))[None],
        }
