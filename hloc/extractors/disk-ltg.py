"""
Code for Using Disk Completed by LightGlue and Compatible for Hloc FrameWork
Mark:
    Just Inherit From built-in Completed in Hloc, only change the descriptors shape
    from (feature_dim, n) to (n, feature_dim)
So, this Script only used when matcher is LightGlue
"""

import torch
import torch.nn.functional as F
from .disk import DISK


class DISKLtg(DISK):
    def _forward(self, data: dict) -> dict:
        image = data['image']
        # make sure that the dimensions of the image are multiple of 16
        orig_h, orig_w = image.shape[-2:]
        new_h = round(orig_h / 16) * 16
        new_w = round(orig_w / 16) * 16
        image = F.pad(image, (0, new_w - orig_w, 0, new_h - orig_h))

        batched_features = self.extract(image)

        assert(len(batched_features) == 1)
        features = batched_features[0]

        # filter points detected in the padded areas
        kpts = features.kp
        valid = torch.all(kpts <= kpts.new_tensor([orig_w, orig_h]) - 1, 1)
        kpts = kpts[valid]
        descriptors = features.desc[valid]
        scores = features.kp_logp[valid]

        # order the keypoints
        indices = torch.argsort(scores, descending=True)
        kpts = kpts[indices]
        descriptors = descriptors[indices]
        scores = scores[indices]

        return {
            'keypoints': kpts[None],
            'descriptors': descriptors[None],
            'scores': scores[None],
        }
