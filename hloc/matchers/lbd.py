import logging
import sys
from pathlib import Path
import torch
import numpy as np
import pytlbd
from ..utils.base_model import BaseModel


class LBD_Heuristic(BaseModel):
    default_conf = {
        'max_n_lines': 512,
    }
    required_inputs = [
        'lines0', 'descriptors0',
        'lines1', 'descriptors1',
    ]

    def _init(self, conf):
        self.max_n_lines = conf['max_n_lines']

    @staticmethod
    def to_multiscale_lines(lines):
        ms_lines = []
        for l in lines.reshape(-1, 4):
            ll = np.append(l, [0, np.linalg.norm(l[:2] - l[2:4])])
            ms_lines.append([(0, ll)] + [(i, ll / (i * np.sqrt(2))) for i in range(1, 5)])
        return ms_lines


    def _forward(self, data):
        lines0 = data['lines0'].detach().cpu().numpy().squeeze()
        lines1 = data['lines1'].detach().cpu().numpy().squeeze()
        ms_lines0 = self.to_multiscale_lines(lines0)
        ms_lines1 = self.to_multiscale_lines(lines1)

        matches0 = torch.full((len(ms_lines0),), -1, dtype=torch.long, device=data['lines0'].device)
        assert len(lines0) <= 1024 and len(lines1) <= 1024, 'Too many lines for LBD Heuristic :('
        # assert len(lines0) <= 1024 and len(lines1) <= 1024, 'Too many lines for LBD Heuristic :('
        # Assume that lines are sorted by relevance
        try:
            matches = pytlbd.lbd_matching_multiscale(
                ms_lines0[:self.max_n_lines], ms_lines1[:self.max_n_lines],
                list(data['descriptors0'].detach().cpu().numpy().squeeze()[:self.max_n_lines]),
                list(data['descriptors1'].detach().cpu().numpy().squeeze()[:self.max_n_lines]))
            matches_ij = np.array(matches)
            matches0[matches_ij[:, 0]] = matches0.new_tensor(matches_ij[:, 1])
        except RuntimeError as e:
            logging.warning(f'Detected exception during matching:\n {e}')

        return {
            'matches0': matches0.unsqueeze(0),
            'matching_scores0': matches0.new_ones((len(lines0),)).unsqueeze(0),
        }
