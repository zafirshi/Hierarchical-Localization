import torch
from .superpoint import SuperPoint


class SuperPointLtg(SuperPoint):
    def _forward(self, data: dict) -> dict:
        # get original output format which is same to spp repo
        original_output = self.net(data)
        # add batch dimension and transpose last from (256, n) to (n, 256)
        return {
            'keypoints': torch.stack(original_output['keypoints'], 0),                          # shape:(n, 2)
            'keypoint_scores': torch.stack(original_output['scores'], 0),                       # shape:(n,)
            'descriptors': torch.stack(original_output['descriptors'], 0).transpose(-1, -2),    # shape:(n, 256)
        }
