import sys

import cv2
import numpy as np
import onnxruntime as ort
import torch
from pathlib import Path

from third_party.ZippyPoint.utils.utils import process_resize
from ..utils.base_model import BaseModel

hfnet_path = Path(__file__).parent / "../../third_party/hfnet"
sys.path.append(str(hfnet_path))


class HfNet(BaseModel):
    default_conf = {
        'keypoint_threshold': 0.005,
        'max_keypoints': -1,
        'input_shape': [640, 480],
        'max_resize': 640,
        'nms_radius': 3,
    }
    required_inputs = ['image']

    def _init(self, conf):
        # set ort session options
        sess_options = ort.SessionOptions()

        sess_options.intra_op_num_threads = 4
        sess_options.inter_op_num_threads = 1
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # load onnx model(and only use CPU)
        onnx_path = hfnet_path / 'model_dynamic.onnx'
        self.onnx_sess = ort.InferenceSession(onnx_path, sess_options, providers=['CPUExecutionProvider'])

        # some ort params
        self.input_name = self.onnx_sess.get_inputs()[0].name
        provider = self.onnx_sess.get_providers()
        print('support provider:   ', provider)

    @staticmethod
    def _load_img(img, resize):
        w, h = img.shape[1], img.shape[0]
        w_new, h_new = process_resize(w, h, resize)
        img = cv2.resize(img, (w_new, h_new), interpolation=cv2.INTER_AREA)
        img = img[None].astype('float32')  # numpy shape: (1, H, W, C)
        return img

    @staticmethod
    def simple_nms(scores, nms_radius: int):
        """ Fast Non-maximum suppression to remove nearby points """
        assert (nms_radius >= 0)

        # numpy to torch
        scores = torch.from_numpy(scores)

        def max_pool(x):
            return torch.nn.functional.max_pool2d(
                x, kernel_size=nms_radius * 2 + 1, stride=1, padding=nms_radius)

        zeros = torch.zeros_like(scores)
        max_mask = scores == max_pool(scores)
        for _ in range(2):
            supp_mask = max_pool(max_mask.float()) > 0
            supp_scores = torch.where(supp_mask, zeros, scores)
            new_max_mask = supp_scores == max_pool(supp_scores)
            max_mask = max_mask | (new_max_mask & (~supp_mask))
        return torch.where(max_mask, scores, zeros).cpu().numpy()

    def _forward(self, data):
        img = data['image']  # Pytorch tensor shape:(B, C, H, W)
        assert img.shape[0] == 1  # inference should run with batch_size=1

        # torch to numpy like cv2.read format
        frame = img[0].cpu().permute(1, 2, 0).numpy() * 255

        if self.conf['max_resize'] is not None:
            frame = self._load_img(frame, [self.conf['max_resize']])
        else:
            # if max_resize:None then use input_shape as hard resize standard (may cause distortion)
            frame = self._load_img(frame, self.conf['input_shape'])

        global_feat, local_feat, score_map = self.onnx_sess.run(None, {self.input_name: frame})

        # squeeze batch_size dimension
        global_feat = global_feat.squeeze()
        local_feat = local_feat.squeeze()

        score_map = self.simple_nms(score_map, self.conf['nms_radius'])
        score_map = score_map.squeeze()



        desc = cv2.resize(local_feat, self.conf['input_shape'])
        non_zero = np.flip(np.argwhere(score_map > self.conf['keypoint_threshold']), 1)
        score_zero = np.expand_dims(score_map[score_map > self.conf['keypoint_threshold']], -1)
        scores = np.concatenate([non_zero, score_zero], axis=1)
        descriptors = desc[score_map > self.conf['keypoint_threshold'], :]

        # select topK key points
        sorted_prob = scores[scores[:, 2].argsort(), :]
        sorted_desc = descriptors[scores[:, 2].argsort(), :]
        start = min(self.conf['max_keypoints'], scores.shape[0])
        selected_points = sorted_prob[-start:, :]
        selected_descriptors = sorted_desc[-start:, :]

        # add a dimension or nest in list or tuple
        return {
            'keypoints': selected_points[:, :2][None],
            'scores': selected_points[:, 2:][None],
            'descriptors': selected_descriptors.T[None],
        }
