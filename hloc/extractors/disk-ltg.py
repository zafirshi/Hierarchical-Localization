"""
Code for Using Disk Completed by LightGlue and Compatible for Hloc FrameWork
Mark:
    Just Inherit From built-in Completed in Hloc, only change the descriptors shape
    from (feature_dim, n) to (n, feature_dim)
So, this Script only used when matcher is LightGlue
"""

import torch
from .disk import DISK


class DISKLtg(DISK):
    def _forward(self, data: dict) -> dict:
        # get original output format which is same to disk repo
        original_output = self.net(data)
        # add batch dimension and transpose last from (256, n) to (n, 256)
        return {
            'keypoints': original_output['keypoints'],                          # shape:(n, 2)
            'keypoint_scores': original_output['scores'],                       # shape:(n,)
            'descriptors': original_output['descriptors'].transpose(-1, -2),    # shape:(n, 256)
        }