#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Edge TPU DeepLabv3 Image segmenation Raspberry Pi camera stream.

    Copyright (c) 2019 Nobuo Tsukamoto

    This software is released under the MIT License.
    See the LICENSE file in the project root for more information.
"""
import argparse
import io
import os
import time

import numpy as np
from PIL import Image

import cv2
from edgetpu.basic.basic_engine import BasicEngine
from utils import visualization as visual
from utils import label_util

WINDOW_NAME = 'Edge TPU Segmentation'

LABEL_NAMES = np.asarray(
    [
        "background",
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "sofa",
        "train",
        "tv",
    ]
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='File path of Tflite model.', required=True)
    parser.add_argument('--width', help='Resolution width.', default=640, type=int)
    parser.add_argument('--height', help='Resolution height.', default=480, type=int)
    parser.add_argument('--nano', help='Works with JETSON Nao and Pi Camera.', action='store_true')
    # parser.add_argument(
    #    '--label', help='File path of label file.', required=True)
    args = parser.parse_args()

    # Initialize window.
    cv2.namedWindow(
        WINDOW_NAME, cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE | cv2.WINDOW_KEEPRATIO
    )
    cv2.moveWindow(WINDOW_NAME, 100, 200)

    # Initialize colormap
    FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
    FULL_COLOR_MAP = label_util.label_to_color_image(FULL_LABEL_MAP)

    # Initialize engine.
    engine = BasicEngine(args.model)
    _, width, height, channels = engine.get_input_tensor_shape()

    if args.nano == True:
        GST_STR = 'nvarguscamerasrc \
            ! video/x-raw(memory:NVMM), width={0:d}, height={1:d}, format=(string)NV12, framerate=(fraction)30/1 \
            ! nvvidconv flip-method=2 !  video/x-raw, width=(int){2:d}, height=(int){3:d}, format=(string)BGRx \
            ! videoconvert \
            ! appsink'.format(args.width, args.height, args.width, args.height)
        cap = cv2.VideoCapture(GST_STR, cv2.CAP_GSTREAMER)

    else:
        cap = cv2.VideoCapture(0)
        cap.set(3, args.width)
        cap.set(4, args.height)

    while(cap.isOpened()):
        _, frame = cap.read()

        start_ms = time.time()

        # Create inpute tensor
        # camera resolution  => input tensor size (513, 513)
        input_buf = cv2.resize(frame, (width, height))
        input_buf = cv2.cvtColor(input_buf, cv2.COLOR_BGR2RGB)
        input_tensor = input_buf.flatten()

        # Run inference
        latency, result = engine.RunInference(input_tensor)

        # Create segmentation map
        seg_map = np.array(result, dtype=np.uint8)
        seg_map = np.reshape(seg_map, (width, height))
        seg_image = label_util.label_to_color_image(seg_map)

        # segmentation map resize 513, 513 => camera resolution
        seg_image = cv2.resize(seg_image, (args.width, args.height))
        im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) // 2 + seg_image // 2
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)

        elapsed_ms = time.time() - start_ms

        # Calc fps.
        fps = 1 / elapsed_ms
        fps_text = '{0:.2f}ms, {1:.2f}fps'.format((elapsed_ms * 1000.0), fps)
        visual.draw_caption(im, (10, 30), fps_text)

        latency_text = 'RunInference latency: {0:.2f}ms'.format(latency)
        visual.draw_caption(im, (10, 60), latency_text)

        # Display image
        cv2.imshow(WINDOW_NAME, im)
        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
            break

        if args.nano != True:
            for i in range(10):
                ret, frame = cap.read()
                
    # When everything done, release the window
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
