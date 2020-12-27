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
import picamera
from picamera.array import PiRGBArray
from PIL import Image
from pycoral.adapters import common, segment
from pycoral.utils.edgetpu import make_interpreter
from utils import label_util
from utils import visualization as visual

WINDOW_NAME = "Edge TPU Segmentation"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="File path of Tflite model.", required=True)
    parser.add_argument("--width", help="Resolution width.", default=640)
    parser.add_argument("--height", help="Resolution height.", default=480)
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

    resolution_width = args.width
    rezolution_height = args.height
    with picamera.PiCamera() as camera:

        camera.resolution = (resolution_width, rezolution_height)
        camera.framerate = 30
        rawCapture = PiRGBArray(camera)

        # allow the camera to warmup
        time.sleep(0.1)

        try:
            for frame in camera.capture_continuous(
                rawCapture, format="rgb", use_video_port=True
            ):

                rawCapture.truncate(0)

                image = frame.array
                im = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                start = time.perf_counter()

                # Create inpute tensor
                # camera resolution (640, 480) => input tensor size (513, 513)
                _, scale = common.set_resized_input(
                    interpreter, (resolution_width, rezolution_height), lambda size: cv2.resize(image, size)
                )
                # Run inference.
                interpreter.invoke()

                elapsed_ms = (time.perf_counter() - start) * 1000
                
                # Create segmentation map
                result = segment.get_output(interpreter)
                seg_map = result[:height, :width]
                seg_image = label_util.label_to_color_image(colormap, seg_map)

                # segmentation map resize 513, 513 => camera resolution(640, 480)
                seg_image = cv2.resize(seg_image, (resolution_width, rezolution_height))
                out_image = image // 2 + seg_image // 2
                im = cv2.cvtColor(out_image, cv2.COLOR_RGB2BGR)  # display image

                # Calc fps.
                fps = 1000. / elapsed_ms
                fps_text = "{0:.2f}ms, {1:.2f}fps".format(elapsed_ms, fps)
                visual.draw_caption(im, (10, 30), fps_text)

                # Display image
                cv2.imshow(WINDOW_NAME, im)
                key = cv2.waitKey(10) & 0xFF
                if key == ord("q"):
                    break

        finally:
            camera.stop_preview()

    # When everything done, release the window
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
