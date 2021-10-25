import sys
from pathlib import Path
import pytlbd
from ..utils.base_model import BaseModel


class LBD_Heuristic(BaseModel):
    default_conf = {

    }
    required_inputs = [
        'line_segments_ms0', 'descriptors0',
        'line_segments_ms1', 'descriptors1',        
    ]

    def _init(self, conf):
        pass

    def _forward(self, data):

        #matches = pytlbd.lbd_matching_multiscale(data['line_segments_ms0'], data['line_segments_ms1'], data['descriptors0'], data['descriptors0'])
        # TODO: Figure out how to store multi-scale descriptors so we can use LBD heuristic here...

        return {
            #'matches0': matches0,
            #'matching_scores0': scores0,
        }
