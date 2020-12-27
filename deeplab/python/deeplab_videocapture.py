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

import cv2
import numpy as np
from pycoral.adapters import common, segment
from pycoral.utils.edgetpu import make_interpreter
from utils import label_util
from utils import visualization as visual

WINDOW_NAME = "Edge TPU Segmentation"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="File path of Tflite model.", required=True)
    parser.add_argument("--width", help="Resolution width.", default=640, type=int)
    parser.add_argument("--height", help="Resolution height.", default=480, type=int)
    parser.add_argument(
        "--nano", help="Works with JETSON Nao and Pi Camera.", action="store_true"
    )
    args = parser.parse_args()

    # Initialize window.
    cv2.namedWindow(
        WINDOW_NAME, cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE | cv2.WINDOW_KEEPRATIO
    )
    cv2.moveWindow(WINDOW_NAME, 100, 200)

    # Initialize colormap
    colormap = label_util.create_pascal_label_colormap()

    # Initialize engine.
    interpreter = make_interpreter(args.model)
    interpreter.allocate_tensors()
    width, height = common.input_size(interpreter)

    if args.nano == True:
        GST_STR = "nvarguscamerasrc \
            ! video/x-raw(memory:NVMM), width={0:d}, height={1:d}, format=(string)NV12, framerate=(fraction)30/1 \
            ! nvvidconv flip-method=2 !  video/x-raw, width=(int){2:d}, height=(int){3:d}, format=(string)BGRx \
            ! videoconvert \
            ! appsink".format(
            args.width, args.height, args.width, args.height
        )
        cap = cv2.VideoCapture(GST_STR, cv2.CAP_GSTREAMER)

    else:
        cap = cv2.VideoCapture(0)
        cap.set(3, args.width)
        cap.set(4, args.height)

    cap_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    while cap.isOpened():
        _, frame = cap.read()

        start = time.perf_counter()

        # Create inpute tensor
        # camera resolution  => input tensor size (513, 513)
        input_buf = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        _, scale = common.set_resized_input(
            interpreter, (cap_width, cap_height), lambda size: cv2.resize(input_buf, size)
        )

        # Run inference
        interpreter.invoke()

        elapsed_ms = (time.perf_counter() - start) * 1000

        # Create segmentation map
        result = segment.get_output(interpreter)
        seg_map = result[:height, :width]
        seg_image = label_util.label_to_color_image(colormap, seg_map)

        # segmentation map resize 513, 513 => camera resolution
        seg_image = cv2.resize(seg_image, (args.width, args.height))
        im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) // 2 + seg_image // 2
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)

        # Calc fps.
        fps = 1000. / elapsed_ms
        fps_text = "{0:.2f}ms, {1:.2f}fps".format(elapsed_ms, fps)
        visual.draw_caption(im, (10, 30), fps_text)

        # Display image
        cv2.imshow(WINDOW_NAME, im)
        key = cv2.waitKey(10) & 0xFF
        if key == ord("q"):
            break

        if args.nano != True:
            for i in range(10):
                ret, frame = cap.read()

    # When everything done, release the window
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
