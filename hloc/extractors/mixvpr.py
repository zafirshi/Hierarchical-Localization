import sys
from pathlib import Path

import torch
import torchvision.transforms as tvf

from ..utils.base_model import BaseModel

sys.path.append(str(Path(__file__).parent / '../../third_party/MixVPR'))
from main import VPRModel


class MixVPR(BaseModel):
    default_conf = {
        'backbone_arch': 'resnet50',
        'layers_to_crop': [4],
        'agg_arch': 'MixVPR',
        # output_dim = 4096 (out_channel * mix_depth)
        'agg_config': {'in_channels': 1024,
                       'in_h': 20,
                       'in_w': 20,
                       'out_channels': 1024,
                       'mix_depth': 4,
                       'mlp_ratio': 1,
                       'out_rows': 4}
    }
    required_inputs = ['image']

    def _init(self, conf):
        self.net = VPRModel(
            backbone_arch=conf['backbone_arch'],
            layers_to_crop=conf['layers_to_crop'],
            agg_arch=conf['agg_arch'],
            agg_config=conf['agg_config']
        )

        pre_trained_path = str(Path(__file__).parent / '../../third_party')
        # download ckpt and put into this path
        state_dict = torch.load(pre_trained_path+'/MixVPR/ckpts/resnet50_MixVPR_4096_channels(1024)_rows(4).ckpt')
        self.net.load_state_dict(state_dict)
        self.net.eval()

        # Resize : Note that images must be resized to 320x320
        self.pre_resize = tvf.Resize((320, 320), interpolation=tvf.InterpolationMode.BILINEAR)
        # norm : imagenet mean std, for vit should change
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.norm_rgb = tvf.Normalize(mean=mean, std=std)
        # pre_process
        self.pre_process = tvf.Compose([
            self.pre_resize,
            self.norm_rgb,
        ])

    def _forward(self, data):
        image = self.pre_process(data['image'])
        desc = self.net(image)
        return {
            'global_descriptor': desc,
        }
