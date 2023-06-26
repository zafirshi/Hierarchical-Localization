import math
import os
from collections import OrderedDict
from pathlib import Path
from typing import Dict

import numpy as np
import cv2 as cv
import torch
from skimage import io
import random

from scipy.spatial.transform import Rotation as Rotation


def read_pose_data(file_name: Path) -> Dict:
    """
    Expects path to file with one pose per line.
    Input pose is expected to map world to camera coordinates.
    Output pose maps camera to world coordinates.
    Pose format: file qw qx qy qz tx ty tz (f)
    Return dictionary that maps a file name to a tuple of (4x4 pose, focal_length)
    Sets focal_length to None if not contained in file.
    """

    with open(file_name, "r") as f:
        pose_data = f.readlines()

    # create a dict from the poses with file name as key
    pose_dict = {}
    for pose_string in pose_data:

        pose_string = pose_string.split()
        file_name = pose_string[0]

        pose_q = np.array(pose_string[1:5])
        pose_q = np.array([pose_q[1], pose_q[2], pose_q[3], pose_q[0]])
        pose_t = np.array(pose_string[5:8])
        pose_R = Rotation.from_quat(pose_q).as_matrix()

        pose_4x4 = np.identity(4)
        pose_4x4[0:3, 0:3] = pose_R
        pose_4x4[0:3, 3] = pose_t

        # convert world->cam to cam->world for evaluation
        pose_4x4 = np.linalg.inv(pose_4x4)

        if len(pose_string) > 8:
            focal_length = float(pose_string[8])
        else:
            focal_length = None

        pose_dict[file_name] = (pose_4x4, focal_length)

    return pose_dict


def compute_error_max_rot_trans(pgt_pose, est_pose):
    """
    Compute the pose error.
    Expects poses to map camera coordinate to world coordinates.
    """

    # calculate pose errors
    t_err = float(np.linalg.norm(pgt_pose[0:3, 3] - est_pose[0:3, 3]))  # unit in meters

    r_err = est_pose[0:3, 0:3] @ np.transpose(pgt_pose[0:3, 0:3])
    r_err = cv.Rodrigues(r_err)[0]
    r_err = np.linalg.norm(r_err) * 180 / math.pi

    return r_err, t_err


def evaluate_pose_err(gt_path: Path, pred_path: Path,
                      error_max_images=-1,
                      error_threshold=5):
    # load pseudo ground truth
    pgt_poses = read_pose_data(gt_path)

    if 0 < error_max_images < len(pgt_poses):
        keys = random.sample(pgt_poses.keys(), error_max_images)
        pgt_poses = {k: pgt_poses[k] for k in keys}

    # load estimated poses
    est_poses = read_pose_data(pred_path)

    # check pred results
    assert len(est_poses) == len(pgt_poses), f'est_poses should get {len(pgt_poses)} but get {len(est_poses)}'

    # main evaluation loop
    errors = np.ndarray((len(pgt_poses), 2))

    for i, query_file in enumerate(pgt_poses):

        try:
            pgt_pose, rgb_focal_length = pgt_poses[query_file]
            est_pose, _ = est_poses[query_file]

            errors[i] = compute_error_max_rot_trans(pgt_pose, est_pose)

        except KeyError:
            # catching the case that an algorithm did not provide an estimate
            errors[i] = math.inf

    # follow hloc recall for  (0.25m, 2°) / (0.5m, 5°) / (5m, 10°)
    recall_array = np.zeros(3, )

    for i, each_err in enumerate(errors):
        if 0 < each_err[0] <= 2 and 0 < each_err[1] <= 0.25:
            recall_array += 1
        elif each_err[0] <= 5 and each_err[1] <= 0.5:
            recall_array[1:] += 1
        elif each_err[0] <= 10 and each_err[1] <= 5:
            recall_array[2:] += 1
        else:
            print(f'Query img {list(est_poses)[i]} err is {each_err}')

    recall_array = recall_array / errors.shape[0]

    print(f"Loc recall in (0.25m, 2°) / (0.5m, 5°) / (5m, 10°) is: "
          f"{recall_array[0] * 100:5.1f}/{recall_array[1] * 100:5.1f}/{recall_array[2] * 100:5.1f}")


if __name__ == '__main__':
    gt_pose_path = Path('./datasets/chunxiroad/queries_gt_pose.txt')
    query_pre_path = Path(
        'outputs/ChunXiRoad/CPU_rz640_topk30_wMR_SGV_nDC6/ChunXiRoad_hloc_zippypoint_aachen+zippypoint-matcher_netvlad30.txt')
    evaluate_pose_err(gt_pose_path, query_pre_path)
