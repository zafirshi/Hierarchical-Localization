import argparse
import os

import torch
from pathlib import Path
from typing import Dict, List, Union, Optional, Tuple, Iterable
import h5py
from types import SimpleNamespace
import cv2
import numpy as np
from tqdm import tqdm
import pprint
import collections.abc as collections
import PIL.Image

from . import extractors, logger
from .utils.base_model import dynamic_load
from .utils.parsers import parse_image_lists
from .utils.io import read_image, list_h5_names

from .utils.viz import plot_keypoints, plot_images, save_plot
import matplotlib.pyplot as plt

'''
A set of standard configurations that can be directly selected from the command
line using their name. Each is a dictionary with the following entries:
    - output: the name of the feature file that will be generated.
    - model: the model configuration, as passed to a feature extractor.
    - preprocessing: how to preprocess the images read from disk.
'''
confs = {
    'superpoint_aachen': {
        'output': 'feats-superpoint-n4096-r1024',
        'model': {
            'name': 'superpoint',
            'nms_radius': 3,
            'max_keypoints': 4096,
        },
        'preprocessing': {
            'grayscale': True,
            'resize_max': 1024,
        },
    },
    # Resize images to 1600px even if they are originally smaller.
    # Improves the keypoint localization if the images are of good quality.
    'superpoint_max': {
        'output': 'feats-superpoint-n4096-rmax1600',
        'model': {
            'name': 'superpoint',
            'nms_radius': 3,
            'max_keypoints': 4096,
        },
        'preprocessing': {
            'grayscale': True,
            'resize_max': 1600,
            'resize_force': True,
        },
    },
    'superpoint_inloc': {
        'output': 'feats-superpoint-n4096-r1600',
        'model': {
            'name': 'superpoint',
            'nms_radius': 4,
            'max_keypoints': 4096,
        },
        'preprocessing': {
            'grayscale': True,
            'resize_max': 1600,
        },
    },
    'r2d2': {
        'output': 'feats-r2d2-n5000-r1024',
        'model': {
            'name': 'r2d2',
            'max_keypoints': 5000,
        },
        'preprocessing': {
            'grayscale': False,
            'resize_max': 1024,
        },
    },
    'd2net-ss': {
        'output': 'feats-d2net-ss',
        'model': {
            'name': 'd2net',
            'multiscale': False,
        },
        'preprocessing': {
            'grayscale': False,
            'resize_max': 1600,
        },
    },
    'sift': {
        'output': 'feats-sift',
        'model': {
            'name': 'dog'
        },
        'preprocessing': {
            'grayscale': True,
            'resize_max': 1600,
        },
    },
    'sosnet': {
        'output': 'feats-sosnet',
        'model': {
            'name': 'dog',
            'descriptor': 'sosnet'
        },
        'preprocessing': {
            'grayscale': True,
            'resize_max': 1600,
        },
    },
    'disk': {
        'output': 'feats-disk',
        'model': {
            'name': 'disk',
            'max_keypoints': 5000,
        },
        'preprocessing': {
            'grayscale': False,
            'resize_max': 1600,
        },
    },
    # new added local descriptors
    'zippypoint_aachen': {
        'output': 'feats-zippypoint-n4096-r640',
        'model': {
            'name': 'zippypoint',
            'nms_radius': 3,
            'keypoint_threshold': 0.0001,
            'max_keypoints': 4096,
            'max_resize': 640,  # 1024, 768
        },
        'resize_up': True,
        'preprocessing': {
            'grayscale': False,
        },
    },
    'zippypoint_aachen_n4096_r1024': {
        'output': 'feats-zippypoint-n4096-r640',
        'model': {
            'name': 'zippypoint',
            'nms_radius': 3,
            'keypoint_threshold': 0.0001,
            'max_keypoints': 4096,
            'max_resize': 1024,  # 1024, 768
        },
        'resize_up': True,
        'preprocessing': {
            'grayscale': False,
        },
    },
    # hfnet_onnx config
    'hfnet_onnx': {
        'output': 'feats-hfnet-n4096-r640',
        'model': {
            'name': 'hfnet',
            'keypoint_threshold': 0.005,    # code 0.015
            'max_keypoints': 4096,          # code 2000
            'input_shape': [640, 480],
            'max_resize': None,  # use default input_shape W,H: 640, 480
        },
        'resize_up': True,
        'preprocessing': {
            'grayscale': False,
        },
    },
    # Global descriptors
    'dir': {
        'output': 'global-feats-dir',
        'model': {'name': 'dir'},
        'preprocessing': {'resize_max': 1024},
    },
    'netvlad': {
        'output': 'global-feats-netvlad',
        'model': {'name': 'netvlad'},
        'preprocessing': {'resize_max': 1024},
    },
    'openibl': {
        'output': 'global-feats-openibl',
        'model': {'name': 'openibl'},
        'preprocessing': {'resize_max': 1024},
    },
    'cosplace': {
        'output': 'global-feats-cosplace',
        'model': {'name': 'cosplace'},
        'preprocessing': {'resize_max': 1024},
    },
    # Add mixvpr
    'mixvpr': {
        'output': 'global-feats-mixvpr',
        'model': {'name': 'mixvpr'},
        'preprocessing': {'resize_max': 1024},
    },
}


def resize_image(image, size, interp):
    if interp.startswith('cv2_'):
        interp = getattr(cv2, 'INTER_' + interp[len('cv2_'):].upper())
        h, w = image.shape[:2]
        if interp == cv2.INTER_AREA and (w < size[0] or h < size[1]):
            interp = cv2.INTER_LINEAR
        resized = cv2.resize(image, size, interpolation=interp)
    elif interp.startswith('pil_'):
        interp = getattr(PIL.Image, interp[len('pil_'):].upper())
        resized = PIL.Image.fromarray(image.astype(np.uint8))
        resized = resized.resize(size, resample=interp)
        resized = np.asarray(resized, dtype=image.dtype)
    else:
        raise ValueError(
            f'Unknown interpolation {interp}.')
    return resized


class ImageDataset(torch.utils.data.Dataset):
    default_conf = {
        'globs': ['*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG'],
        'grayscale': False,
        'resize_max': None,
        'resize_force': False,
        'interpolation': 'cv2_area',  # pil_linear is more accurate but slower
    }

    def __init__(self, root, conf, paths=None):
        self.conf = conf = SimpleNamespace(**{**self.default_conf, **conf})
        self.root = root

        if paths is None:
            paths = []
            for g in conf.globs:
                paths += list(Path(root).glob('**/' + g))
            if len(paths) == 0:
                raise ValueError(f'Could not find any image in root: {root}.')
            paths = sorted(list(set(paths)))
            self.names = [i.relative_to(root).as_posix() for i in paths]
            logger.info(f'Found {len(self.names)} images in root {root}.')
        else:
            if isinstance(paths, (Path, str)):
                self.names = parse_image_lists(paths)
            elif isinstance(paths, collections.Iterable):
                self.names = [p.as_posix() if isinstance(p, Path) else p
                              for p in paths]
            else:
                raise ValueError(f'Unknown format for path argument {paths}.')

            for name in self.names:
                if not (root / name).exists():
                    raise ValueError(
                        f'Image {name} does not exists in root: {root}.')

    def __getitem__(self, idx):
        name = self.names[idx]
        image = read_image(self.root / name, self.conf.grayscale)
        image = image.astype(np.float32)
        size = image.shape[:2][::-1]

        if self.conf.resize_max and (self.conf.resize_force
                                     or max(size) > self.conf.resize_max):
            scale = self.conf.resize_max / max(size)
            size_new = tuple(int(round(x * scale)) for x in size)
            image = resize_image(image, size_new, self.conf.interpolation)

        if self.conf.grayscale:
            image = image[None]
        else:
            image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
        image = image / 255.

        data = {
            'image': image,
            'original_size': np.array(size),
        }
        return data

    def __len__(self):
        return len(self.names)


def debug_viz(image_dir, image_name, keypoints,
              save_dir: Path,
              dpi=75):
    # Note: comment one now is used built_in interface, uncomment one is Self-implemented method
    plt.imshow(read_image(image_dir / image_name))
    # plot_images([read_image(image_dir / image_name)], dpi=dpi)

    plt.scatter(keypoints[:, 0], keypoints[:, 1], c='red', s=4, linewidths=0)
    # plot_keypoints([keypoints], colors='red', ps=4)

    # os.makedirs(save_dir / 'db', exist_ok=True)
    plt.savefig(save_dir / image_name, bbox_inches='tight', pad_inches=0)
    plt.close()
    # save_plot(save_dir / image_name)


@torch.no_grad()
def main(conf: Dict,
         image_dir: Path,
         export_dir: Optional[Path] = None,
         as_half: bool = True,
         image_list: Optional[Union[Path, List[str]]] = None,
         feature_path: Optional[Path] = None,
         overwrite: bool = False) -> Path:
    logger.info('Extracting local features with configuration:'
                f'\n{pprint.pformat(conf)}')

    dataset = ImageDataset(image_dir, conf['preprocessing'], image_list)
    if feature_path is None:
        feature_path = Path(export_dir, conf['output'] + '.h5')
    feature_path.parent.mkdir(exist_ok=True, parents=True)
    skip_names = set(list_h5_names(feature_path)
                     if feature_path.exists() and not overwrite else ())
    dataset.names = [n for n in dataset.names if n not in skip_names]
    if len(dataset.names) == 0:
        logger.info('Skipping the extraction.')
        return feature_path

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Note: Add extractor model here
    Model = dynamic_load(extractors, conf['model']['name'])
    model = Model(conf['model']).eval().to(device)

    loader = torch.utils.data.DataLoader(
        dataset, num_workers=1, shuffle=False, pin_memory=True)
    for idx, data in enumerate(tqdm(loader)):
        name = dataset.names[idx]
        pred = model({'image': data['image'].to(device, non_blocking=True)})
        # Note: value in pred should nest in list or tuple
        for k, v in pred.items():
            assert isinstance(v, Tuple or List) and len(v) == 1, 'value in pred should nest in list or tuple'
            if isinstance(v[0], torch.Tensor):
                pred = {k: v[0].cpu().numpy()}
            elif isinstance(v[0], np.ndarray):
                pred = {k: v[0]}
            else:
                raise ValueError(f'Value returned from extractors should be ndarray or tensor, but get {type(v[0])}')

        pred['image_size'] = original_size = data['original_size'][0].numpy()  # WxH
        if 'keypoints' in pred:
            # if-condition to choose whether use hloc pre-process resize
            # Note: in conf should omit resize_max key, instead set resize_up key
            if 'resize_up' in conf and conf['resize_up']:
                resize_factor = conf['model']['max_resize'] / max(original_size)
                _w, _h = original_size
                size = np.array([int(round(_w * resize_factor)), int(round(_h * resize_factor))])  # WxH
            else:
                size = np.array(data['image'].shape[-2:][::-1])  # WxH
            scales = (original_size / size).astype(np.float32)
            # TODO: check how this +/- 0.5 whether matter
            pred['keypoints'] = (pred['keypoints'] + .5) * scales[None] - .5

            # ADD visual to look point location
            # debug_viz(image_dir, name, pred['keypoints'], save_dir=export_dir / 'viz')

            if 'scales' in pred:
                pred['scales'] *= scales.mean()
            # add keypoint uncertainties scaled to the original resolution
            uncertainty = getattr(model, 'detection_noise', 1) * scales.mean()

        # Note: save storage: half pred[k] from float32 -> float16
        if as_half:
            for k in pred:
                dt = pred[k].dtype
                if (dt == np.float32) and (dt != np.float16):
                    pred[k] = pred[k].astype(np.float16)

        with h5py.File(str(feature_path), 'a', libver='latest') as fd:
            try:
                if name in fd:
                    del fd[name]
                grp = fd.create_group(name)
                for k, v in pred.items():
                    grp.create_dataset(k, data=v)
                if 'keypoints' in pred:
                    grp['keypoints'].attrs['uncertainty'] = uncertainty
            except OSError as error:
                if 'No space left on device' in error.args[0]:
                    logger.error(
                        'Out of disk space: storing features on disk can take '
                        'significant space, did you enable the as_half flag?')
                    del grp, fd[name]
                raise error

        del pred

    logger.info('Finished exporting features.')
    return feature_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=Path, required=True)
    parser.add_argument('--export_dir', type=Path, required=True)
    parser.add_argument('--conf', type=str, default='superpoint_aachen',
                        choices=list(confs.keys()))
    parser.add_argument('--as_half', action='store_true')
    parser.add_argument('--image_list', type=Path)
    parser.add_argument('--feature_path', type=Path)
    args = parser.parse_args()
    main(confs[args.conf], args.image_dir, args.export_dir, args.as_half)
