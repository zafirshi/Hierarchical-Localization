import os
from pathlib import Path
from pprint import pformat
import argparse
import yaml

from hloc import extract_features, match_features, triangulation
from hloc import pairs_from_covisibility, pairs_from_retrieval, localize_sfm

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=Path, default='datasets/aachen_v1.1',
                    help='Path to the dataset, default: %(default)s')
parser.add_argument('--outputs', type=Path, default='outputs/aachen_v1.1',
                    help='Path to the output directory, default: %(default)s')
parser.add_argument('--num_covis', type=int, default=20,
                    help='Number of image pairs for SfM, default: %(default)s')
parser.add_argument('--num_loc', type=int, default=50,
                    help='Number of image pairs for loc, default: %(default)s')
parser.add_argument('--feature_conf', type=str, default='zippypoint_aachen',
                    # silk_aachen | superpoint_aachen | zippypoint_aachen
                    help="Local feature extractor which completed in extract_features' confs dict")
parser.add_argument('--retrieval_conf', type=str, default='netvlad',
                    help="Retrival Method completed in extract_features' confs dict")
parser.add_argument('--matcher_conf', type=str, default='zippypoint-matcher',
                    help="Matcher completed in match_features' confs dict")
parser.add_argument('--skip_geometric_verification', type=bool, default=False,
                    help="whether skip_geometric_verification when triangulation(skip would be fast but inaccurate)")
args = parser.parse_args()

# Setup the paths
dataset = args.dataset
images = dataset / 'images/images_upright/'
sift_sfm = dataset / '3D-models/aachen_v_1_1'

outputs = args.outputs  # where everything will be saved
reference_sfm = outputs / f'sfm_{args.feature_conf}+{args.matcher_conf}'  # the SfM model we will build
sfm_pairs = outputs / f'pairs-db-covis{args.num_covis}.txt'  # top-k most covisible in SIFT model
loc_pairs = outputs / f'pairs-query-{args.retrieval_conf}{args.num_loc}.txt'  # top-k retrieved by Netvlad
results = outputs / f'Aachen_hloc_{args.feature_conf}+{args.matcher_conf}_{args.retrieval_conf}{args.num_loc}.txt'

# list the standard configurations available
# print(f'Configs for feature extractors:\n{pformat(extract_features.confs)}')
# print(f'Configs for feature matchers:\n{pformat(match_features.confs)}')

# pick one of the configurations for extraction and matching
retrieval_conf = extract_features.confs[args.retrieval_conf]
feature_conf = extract_features.confs[args.feature_conf]
matcher_conf = match_features.confs[args.matcher_conf]

# save configs in yaml file
cfgs = {
    'Extractor:' + args.feature_conf: feature_conf,
    'Matcher:' + args.matcher_conf: matcher_conf,
    'Retrieval:' + args.retrieval_conf: retrieval_conf
}
os.makedirs(outputs, exist_ok=True)
with open(Path(outputs / 'config.yml'), 'w') as yaml_file:
    yaml.dump(cfgs, yaml_file, default_flow_style=False)

features = extract_features.main(feature_conf, images, outputs)

pairs_from_covisibility.main(
    sift_sfm, sfm_pairs, num_matched=args.num_covis)
sfm_matches = match_features.main(
    matcher_conf, sfm_pairs, feature_conf['output'], outputs, image_dir=images)

triangulation.main(
    reference_sfm,
    sift_sfm,
    images,
    sfm_pairs,
    features,
    sfm_matches,
    skip_geometric_verification=args.skip_geometric_verification,
)

global_descriptors = extract_features.main(retrieval_conf, images, outputs)
pairs_from_retrieval.main(
    global_descriptors, loc_pairs, args.num_loc,
    query_prefix='query', db_model=reference_sfm)
loc_matches = match_features.main(
    matcher_conf, loc_pairs, feature_conf['output'], outputs, image_dir=images)

localize_sfm.main(
    reference_sfm,
    dataset / 'queries/*_time_queries_with_intrinsics.txt',
    loc_pairs,
    features,
    loc_matches,
    results,
    covisibility_clustering=False)  # not required with SuperPoint+SuperGlue
