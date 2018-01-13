import argparse
import logging
import time

import common
import cv2
import numpy as np
from estimator import TfPoseEstimator
from networks import get_graph_path, model_wh

from lifting.prob_model import Prob3dPose
from lifting.draw import plot_pose

logger = logging.getLogger('TfPoseEstimator')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation run')
    # parser.add_argument('--image', type=str, default='/Users/ildoonet/Downloads/me.jpg')
    parser.add_argument('--image', type=str, default='./images/apink2.jpg')
    # parser.add_argument('--model', type=str, default='mobilenet_320x240', help='cmu / mobilenet_320x240')
    parser.add_argument('--model', type=str, default='mobilenet_thin_432x368', help='cmu_640x480 / cmu_640x360 / mobilenet_thin_432x368')
    parser.add_argument('--background', type=str, default=False, help='whether to do background only')
    parser.add_argument('--src_directory', type=str, default='.', help='direcotry to pull from')


    args = parser.parse_args()

    w, h = model_wh(args.model)
    e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))

    # estimate human poses from a single image !
    image = common.read_imgfile(args.image, w, h)
    t = time.time()
    humans = e.inference(image)
    elapsed = time.time() - t

    logger.info('inference image: %s in %.4f seconds.' % (args.image, elapsed))

    image = cv2.imread(args.image, cv2.IMREAD_COLOR)
    image = TfPoseEstimator.draw_humans(image, humans, args.background, imgcopy=False)
    cv2.imshow('tf-pose-estimation result', image)
    cv2.imwrite('../../Desktop/sample2.jpg', image)
    cv2.waitKey()
