import argparse
import os
from typing import Union, Optional, Dict, List, Tuple
from pathlib import Path
import pprint
from queue import Queue
from threading import Thread
from functools import partial
from tqdm import tqdm
import h5py
import torch

from . import matchers, logger
from .utils.base_model import dynamic_load
from .utils.parsers import names_to_pair, names_to_pair_old, parse_retrieval

from .utils.viz import make_matching_plot_fast
from .utils.io import read_image
import cv2

'''
A set of standard configurations that can be directly selected from the command
line using their name. Each is a dictionary with the following entries:
    - output: the name of the match file that will be generated.
    - model: the model configuration, as passed to a feature matcher.
'''
confs = {
    'superglue': {
        'output': 'matches-superglue',
        'model': {
            'name': 'superglue',
            'weights': 'outdoor',
            'sinkhorn_iterations': 50,
        },
    },
    'superglue-fast': {
        'output': 'matches-superglue-it5',
        'model': {
            'name': 'superglue',
            'weights': 'outdoor',
            'sinkhorn_iterations': 5,
        },
    },
    'NN-superpoint': {
        'output': 'matches-NN-mutual-dist.7',
        'model': {
            'name': 'nearest_neighbor',
            'do_mutual_check': True,
            'distance_threshold': 0.7,
        },
    },
    'NN-ratio': {
        'output': 'matches-NN-mutual-ratio.8',
        'model': {
            'name': 'nearest_neighbor',
            'do_mutual_check': True,
            'ratio_threshold': 0.8,
        }
    },
    'NN-mutual': {
        'output': 'matches-NN-mutual',
        'model': {
            'name': 'nearest_neighbor',
            'do_mutual_check': True,
        },
    },
    'adalam': {
        'output': 'matches-adalam',
        'model': {
            'name': 'adalam'
        },
    },
    # Add new matcher
    'zippypoint-matcher': {
        'output': 'matches-zpp',
        'model': {
            'name': 'zpp_matcher',
            'do_mutual_check': True,
            'ratio_threshold': 0.95,
        },
    },
    # refine config
    'zippypoint-matcher.9': {
        'output': 'matches-zpp',
        'model': {
            'name': 'zpp_matcher',
            'do_mutual_check': True,
            'ratio_threshold': 0.9,
        },
    },
    'zippypoint-matcher_wothd': {
        'output': 'matches-zpp',
        'model': {
            'name': 'zpp_matcher',
            'do_mutual_check': True,
        },
    },
}


class WorkQueue():
    def __init__(self, work_fn, num_threads=1):
        self.queue = Queue(num_threads)
        self.threads = [
            Thread(target=self.thread_fn, args=(work_fn,))
            for _ in range(num_threads)
        ]
        for thread in self.threads:
            thread.start()

    def join(self):
        for thread in self.threads:
            self.queue.put(None)
        for thread in self.threads:
            thread.join()

    def thread_fn(self, work_fn):
        item = self.queue.get()
        while item is not None:
            work_fn(item)
            item = self.queue.get()

    def put(self, data):
        self.queue.put(data)


class FeaturePairsDataset(torch.utils.data.Dataset):
    def __init__(self, pairs, feature_path_q, feature_path_r, image_dir):
        self.pairs = pairs
        self.feature_path_q = feature_path_q
        self.feature_path_r = feature_path_r
        self.image_dir = image_dir

    def __getitem__(self, idx):
        name0, name1 = self.pairs[idx]
        data = {}
        with h5py.File(self.feature_path_q, 'r') as fd:
            grp = fd[name0]
            for k, v in grp.items():
                data[k + '0'] = torch.from_numpy(v.__array__()).float()
            # some matchers might expect an image but only use its size
            data['image0'] = torch.empty((1,) + tuple(grp['image_size'])[::-1])
        data['name0'] = Path(name0).stem
        data['image_path0'] = str(Path(self.image_dir, name0))
        with h5py.File(self.feature_path_r, 'r') as fd:
            grp = fd[name1]
            for k, v in grp.items():
                data[k + '1'] = torch.from_numpy(v.__array__()).float()
            data['image1'] = torch.empty((1,) + tuple(grp['image_size'])[::-1])
        data['name1'] = Path(name1).stem
        data['image_path1'] = str(Path(self.image_dir, name1))
        return data

    def __len__(self):
        return len(self.pairs)


def writer_fn(inp, match_path):
    pair, pred = inp
    with h5py.File(str(match_path), 'a', libver='latest') as fd:
        if pair in fd:
            del fd[pair]
        grp = fd.create_group(pair)
        matches = pred['matches0'][0].cpu().short().numpy()
        grp.create_dataset('matches0', data=matches)
        if 'matching_scores0' in pred:
            scores = pred['matching_scores0'][0].cpu().half().numpy()
            grp.create_dataset('matching_scores0', data=scores)


def main(conf: Dict,
         pairs: Path, features: Union[Path, str],
         export_dir: Optional[Path] = None,
         matches: Optional[Path] = None,
         features_ref: Optional[Path] = None,
         image_dir: Optional[Path] = None,
         overwrite: bool = False) -> Path:
    if isinstance(features, Path) or Path(features).exists():
        features_q = features
        if matches is None:
            raise ValueError('Either provide both features and matches as Path'
                             ' or both as names.')
    else:
        if export_dir is None:
            raise ValueError('Provide an export_dir if features is not'
                             f' a file path: {features}.')
        features_q = Path(export_dir, features + '.h5')
        if matches is None:
            matches = Path(
                export_dir, f'{features}_{conf["output"]}_{pairs.stem}.h5')

    if features_ref is None:
        features_ref = features_q
    match_from_paths(conf, pairs, matches, features_q, features_ref, image_dir, overwrite)

    return matches


def find_unique_new_pairs(pairs_all: List[Tuple[str]], match_path: Path = None):
    '''Avoid to recompute duplicates to save time.'''
    pairs = set()
    for i, j in pairs_all:
        if (j, i) not in pairs:
            pairs.add((i, j))
    pairs = list(pairs)
    if match_path is not None and match_path.exists():
        with h5py.File(str(match_path), 'r', libver='latest') as fd:
            pairs_filtered = []
            for i, j in pairs:
                if (names_to_pair(i, j) in fd or
                        names_to_pair(j, i) in fd or
                        names_to_pair_old(i, j) in fd or
                        names_to_pair_old(j, i) in fd):
                    continue
                pairs_filtered.append((i, j))
        return pairs_filtered
    return pairs


def debug_viz(pred, data, output_dir=None):
    """
    data keys:
    'descriptors0', 'image_size0', 'keypoints0', 'scores0', 'image0', 'name0',
    'descriptors1', 'image_size1', 'keypoints1', 'scores1', 'image1', 'name1'
    pred keys:
    'matches0', 'matching_scores0'
    """
    kpts0 = data['keypoints0'][0].cpu().numpy()
    kpts1 = data['keypoints1'][0].cpu().numpy()
    matches = pred['matches0'][0].cpu().numpy()

    valid = matches > -1
    mkpts0 = kpts0[valid]
    mkpts1 = kpts1[matches[valid]]

    text = [
        'Hloc',
        'Keypoints: {} - {}'.format(len(kpts0), len(kpts1)),
        'Matches: {}'.format(len(mkpts0))
    ]
    print(text)

    small_text = [
        'Image Pair: {} - {}'.format(data['name0'][0], data['name1'][0]),
    ]

    # todo: check image_path0 should be string
    image_path0 = data['image_path0'][0]
    image_path1 = data['image_path1'][0]

    # load_image
    image0 = read_image(image_path0, grayscale=False)  # np.array(H, W, 3)
    image1 = read_image(image_path1, grayscale=False)  # np.array(H, W, 3)

    # Note: data['image'] read from .h5 file is actually not the original image just a size()
    # See: FeaturePairsDataset:__getitem__()
    out = make_matching_plot_fast(
        image0, image1, kpts0, kpts1, mkpts0, mkpts1,
        text,
        path=None, show_keypoints=True, small_text=small_text)

    if output_dir is not None:
        stem = 'matches_{}_{}'.format(data['name0'][0], data['name1'][0])
        out_file = str(Path(output_dir / 'viz_pairs', stem + '.png'))
        os.makedirs(Path(output_dir / 'viz_pairs'), exist_ok=True)
        print('\nWriting image to {}'.format(out_file))
        out = out[:, :, ::-1]  # RGB to BGR for imwrite to save
        cv2.imwrite(out_file, out)


@torch.no_grad()
def match_from_paths(conf: Dict,
                     pairs_path: Path,
                     match_path: Path,
                     feature_path_q: Path,
                     feature_path_ref: Path,
                     image_dir: Path,
                     overwrite: bool = False) -> Path:
    logger.info('Matching local features with configuration:'
                f'\n{pprint.pformat(conf)}')

    if not feature_path_q.exists():
        raise FileNotFoundError(f'Query feature file {feature_path_q}.')
    if not feature_path_ref.exists():
        raise FileNotFoundError(f'Reference feature file {feature_path_ref}.')
    match_path.parent.mkdir(exist_ok=True, parents=True)

    assert pairs_path.exists(), pairs_path
    pairs = parse_retrieval(pairs_path)
    pairs = [(q, r) for q, rs in pairs.items() for r in rs]
    pairs = find_unique_new_pairs(pairs, None if overwrite else match_path)
    if len(pairs) == 0:
        logger.info('Skipping the matching.')
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    Model = dynamic_load(matchers, conf['model']['name'])
    model = Model(conf['model']).eval().to(device)

    dataset = FeaturePairsDataset(pairs, feature_path_q, feature_path_ref, image_dir)
    loader = torch.utils.data.DataLoader(
        dataset, num_workers=5, batch_size=1, shuffle=False, pin_memory=True)
    writer_queue = WorkQueue(partial(writer_fn, match_path=match_path), 5)

    for idx, data in enumerate(tqdm(loader, smoothing=.1)):
        data = {k: v if k.startswith('image') or k.startswith('name') or k.startswith('image_path')
                else v.to(device, non_blocking=True) for k, v in data.items()}
        pred = model(data)
        # add match pair visual
        # debug_viz(pred, data, output_dir=Path(pairs_path).parent)
        pair = names_to_pair(*pairs[idx])
        writer_queue.put((pair, pred))
    writer_queue.join()
    logger.info('Finished exporting matches.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pairs', type=Path, required=True)
    parser.add_argument('--export_dir', type=Path)
    parser.add_argument('--features', type=str,
                        default='feats-superpoint-n4096-r1024')
    parser.add_argument('--matches', type=Path)
    parser.add_argument('--conf', type=str, default='superglue',
                        choices=list(confs.keys()))
    args = parser.parse_args()
    main(confs[args.conf], args.pairs, args.features, args.export_dir)
