import argparse
import logging
import time
import os, sys
from natsort import natsorted, ns

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
    parser.add_argument('--image', type=str, default='')
    # parser.add_argument('--model', type=str, default='mobilenet_320x240', help='cmu / mobilenet_320x240')
    parser.add_argument('--model', type=str, default='mobilenet_thin_432x368', help='cmu_640x480 / cmu_640x360 / mobilenet_thin_432x368')
    parser.add_argument('--noback', type=str, default=False, help='whether to do background only')
    parser.add_argument('--src_directory', type=str, default='.', help='direcotry to pull from')


    args = parser.parse_args()

    w, h = model_wh(args.model)
    e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))

    conversion_counter = 0

    # estimate human poses from a single image !
    if args.image:
        image = common.read_imgfile(args.image, w, h)
        t = time.time()
        humans = e.inference(image)
        elapsed = time.time() - t

        logger.info('inference image: %s in %.4f seconds.' % (args.image, elapsed))

        image = cv2.imread(args.image, cv2.IMREAD_COLOR)
        image = TfPoseEstimator.draw_humans(image, humans, args.noback, imgcopy=False)
        img_path, image_filename = os.path.split(args.image)

        if args.noback:
            if not os.path.exists(img_path + '/skeleton-no-background'):
                os.mkdir(img_path + '/skeleton-no-background')

        if not os.path.exists(img_path + '/skeleton'):
            os.mkdir(img_path + '/skeleton')

        image_filename = image_filename.replace('jpg', '')

        if args.noback:
            image_filename = image_filename + 'snb.jpg'
            image_full = os.path.join((img_path + '/skeleton-no-background/' + image_filename))
        else:
            image_filename = image_filename + 'ske.jpg'
            image_full = os.path.join((img_path + '/skeleton/' + image_filename))

        cv2.imwrite(image_full, image)
        cv2.imshow('tf-pose-estimation result', image)
        cv2.waitKey()

    else:

        dir_path_list = os.listdir(args.src_directory)
        images = natsorted(dir_path_list, key=lambda y: y.lower())

        if '.DS_Store' in images:
            images.remove('.DS_Store')

        while conversion_counter < len(images):
            image_path = os.path.join((args.src_directory + images[conversion_counter]))
            image = common.read_imgfile(image_path, w, h)
            t = time.time()
            humans = e.inference(image)
            elapsed = time.time() - t

            logger.info('inference image: %s in %.4f seconds.' % (images[conversion_counter], elapsed))

            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            image = TfPoseEstimator.draw_humans(image, humans, args.noback, imgcopy=False)
            image_name = images[conversion_counter]

            if args.noback:
                if not os.path.exists(args.src_directory + '/skeleton-no-background'):
                    os.mkdir(args.src_directory + '/skeleton-no-background')

            if not os.path.exists(args.src_directory + '/skeleton'):
                os.mkdir(args.src_directory + '/skeleton')

            image_filename = images[conversion_counter].replace('jpg', '')

            if args.noback:
                image_filename = image_filename + 'snb.jpg'
                image_full = os.path.join((args.src_directory + '/skeleton-no-background/' + image_filename))
            else:
                image_filename = image_filename + 'ske.jpg'
                image_full = os.path.join((args.src_directory + '/skeleton/' + image_filename))

            print('[INFO]: Converted Image ' + images[conversion_counter])

            cv2.imwrite(image_full, image)
            conversion_counter += 1
