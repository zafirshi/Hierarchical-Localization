from pathlib import Path

from hloc import extract_features, match_features, reconstruction, visualization, pairs_from_retrieval

images = Path('datasets/chunxiroad/images/db')

outputs = Path('outputs/sfm2/')
sfm_pairs = outputs / 'pairs-netvlad30.txt'
sfm_dir = outputs / 'sfm_zpp-extractor+zpp-matcher'

retrieval_conf = extract_features.confs['netvlad']
feature_conf = extract_features.confs['zippypoint_aachen']
matcher_conf = match_features.confs['zippypoint-matcher']

# Find image pairs via image retrieval
retrieval_path = extract_features.main(retrieval_conf, images, outputs)
pairs_from_retrieval.main(retrieval_path, sfm_pairs, num_matched=30)

# Extract and match local features
feature_path = extract_features.main(feature_conf, images, outputs)
match_path = match_features.main(matcher_conf, sfm_pairs, feature_conf['output'], outputs, image_dir=images)


# # 3D reconstruction: Run COLMAP on the features and matches.
model = reconstruction.main(sfm_dir, images, sfm_pairs, feature_path, match_path)

