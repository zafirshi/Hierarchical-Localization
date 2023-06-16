import sys
from pathlib import Path
import torch
import cv2
import tensorflow as tf

from ..utils.base_model import BaseModel

zippy_point_path = Path(__file__).parent / "../../third_party/ZippyPoint"
sys.path.append(str(zippy_point_path))

from utils.utils import pre_process, process_resize
from models.zippypoint import load_ZippyPoint
from models.postprocessing import PostProcessing


class Zippy_point(BaseModel):
    default_conf = {
        'nms_radius': 4,
        'keypoint_threshold': 0.005,
        'max_keypoints': -1,
        'input_shape': [640, 480],  # dummy_x input shape for model
        'max_resize': 640
    }
    required_inputs = ['image']

    def _init(self, conf):
        model_file = zippy_point_path / 'models/weights'
        _ZippyPoint = load_ZippyPoint(model_file, input_shape=conf['input_shape'])
        self.net = _ZippyPoint

        self.post_processing = PostProcessing(nms_window=conf['nms_radius'],
                                              max_keypoints=conf['max_keypoints'],
                                              keypoint_threshold=conf['keypoint_threshold'])

    @staticmethod
    def _load_img(img, resize):
        w, h = img.shape[1], img.shape[0]
        w_new, h_new = process_resize(w, h, resize)
        img = cv2.resize(img, (w_new, h_new), interpolation=cv2.INTER_AREA)
        return img

    def _forward(self, data):
        img = data['image']  # Pytorch tensor shape:(B, C, H, W)
        assert img.shape[0] == 1  # inference should run with batch_size=1

        # img now is pytorch tensor should transfer to tensorflow tensor
        # shape:(C, H, W) -> (H, W, C) to fit in_built pre-process in zippy_point
        frame = img[0].cpu().permute(1, 2, 0).numpy() * 255
        frame = frame.astype('uint8')

        frame = frame[:, :, ::-1]  # RGB -> BGR, same to zpp project

        frame = self._load_img(frame, [self.conf['max_resize']])
        frame_tensor, img_pad = pre_process(frame)

        scores, keypoints, descriptors = self.net(frame_tensor, False)
        scores, keypoints, descriptors = self.post_processing(scores, keypoints, descriptors)
        # Correct keypoint location given required padding
        keypoints -= tf.constant([img_pad[2][0], img_pad[1][0]], dtype=tf.float32)

        # Switch descriptor from (n, 256) to (256, n) to fit hloc framework
        descriptors[0] = tf.transpose(descriptors[0])

        return {
            'keypoints': keypoints,  # shape:(n, 2)
            'scores': scores,  # shape:(n,)
            'descriptors': descriptors,  # shape:(256,n)
        }
