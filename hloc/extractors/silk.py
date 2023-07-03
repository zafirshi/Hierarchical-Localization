import os
import sys
from pathlib import Path
import torch

from ..utils.base_model import BaseModel

# Note: hard-code path
# sys.path.append('/media/zafirshi/software/Code/silk')
sys.path.append(str(Path(__file__).parent / '../../third_party/silk'))

from silk.backbones.silk.silk import SiLKVGG
from silk.backbones.superpoint.vgg import ParametricVGG
from silk.backbones.silk.silk import from_feature_coords_to_image_coords
from silk.config.model import load_model_from_checkpoint


class SiLK(BaseModel):

    default_conf = {
        'in_channels': 1,
        'silk_threshold': 1.0,
        'silk_top_k': 10000,
        'nms_radius': 0,
        'silk_border': 0,
        'default_outputs': (  # outputs required when running the model
            "dense_positions",
            "normalized_descriptors",
            "probability",
        ),
        'silk_scale_factor': 1.41,  # follow source code not change

        # FIXME: change hard-code path
        'checkpoint_dir': '/media/zafirshi/software/Code/silk/assets/models/silk/coco-rgb-aug.ckpt'
    }
    required_inputs = ['image']

    def _init(self, conf):
        silk_backbone = ParametricVGG(
            use_max_pooling=False,
            padding=0,
            normalization_fn=[torch.nn.BatchNorm2d(i) for i in (64, 64, 128, 128)],
        )

        model = SiLKVGG(
            in_channels=conf['in_channels'],
            backbone=silk_backbone,
            detection_threshold=conf['silk_threshold'],
            detection_top_k=conf['silk_top_k'],
            nms_dist=conf['nms_radius'],
            border_dist=conf['silk_border'],
            default_outputs=conf['default_outputs'],
            descriptor_scale_factor=conf['silk_scale_factor'],
            padding=0,
        )

        device = 'cuda:0'  # fix running device

        self.net = load_model_from_checkpoint(
            model,
            checkpoint_path=conf['checkpoint_dir'],
            state_dict_fn=lambda x: {k[len("_mods.model."):]: v for k, v in x.items()},
            device=device,
            freeze=True,
            eval=True,
        )

    def _forward(self, data):
        # TODO: test interface and del if-condition in silk-flow file
        returned_tuple = self.net(data)

        # Change Tuple to dict aiming to adopt hloc framework
        out_dict = {}
        for k, v in zip(self.conf['default_outputs'], returned_tuple):
            # Interface legal format SEE: ./third_party/silk/doc/usage/interface.md
            if k == 'sparse_descriptors':   # Original Shape: (n, 128)
                out_dict['descriptors'] = tuple([v[0].permute(1, 0)])  # Shape (n, 128) -> (128, n)
            elif k == 'sparse_positions':   # Original Shape: (n, 3)
                one_sparse_position = v[0][:, :-1]
                # feature -> image coords
                one_sparse_position = from_feature_coords_to_image_coords(self.net, one_sparse_position)
                # flip h,w order in dim1 (similar to spp -> Convert (h, w) to (x, y))
                out_dict['keypoints'] = tuple([torch.flip(one_sparse_position, [1])])     # Shape (n, 3) -> (n, 2)

                out_dict['scores'] = tuple([v[0][:, -1]])         # Shape (n, 3) -> (n,)
            else:
                raise KeyError(f"SiLK default output should be in ['features', 'logits', 'probability', "
                               f"'raw_descriptors', 'sparse_positions', 'sparse_descriptors',"
                               f"'dense_positions', 'dense_descriptors'] but get{k}")
        return out_dict
