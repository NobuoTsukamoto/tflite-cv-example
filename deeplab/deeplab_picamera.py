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
import picamera
from edgetpu.basic.basic_engine import BasicEngine
from picamera.array import PiRGBArray
from utils import visualization as visual

WINDOW_NAME = "Edge TPU Segmentation"

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


def create_pascal_label_colormap():
    """ Creates a label colormap used in PASCAL VOC segmentation benchmark.

    Returns:
        A Colormap for visualizing segmentation results.
    """
    colormap = np.zeros((256, 3), dtype=np.uint8)
    ind = np.arange(256, dtype=np.uint8)

    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:, channel] |= ((ind >> channel) & 1) << shift
        ind >>= 3
    return colormap


def label_to_color_image(label):
    """ Adds color defined by the dataset colormap to the label.

    Args:
        label: A 2D array with integer type, storing the segmentation label.

    Returns:
        result: A 2D array with floating type. The element of the array
            is the color indexed by the corresponding element in the input label
            to the PASCAL color map.

    Raises:
        ValueError: If label is not of rank 2 or its value is larger than color
            map maximum entry.
    """
    if label.ndim != 2:
        raise ValueError("Expect 2-D input label")

    colormap = create_pascal_label_colormap()

    if np.max(label) >= len(colormap):
        raise ValueError("label value too large.")

    return colormap[label]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="File path of Tflite model.", required=True)
    parser.add_argument("--width", help="Resolution width.", default=640)
    parser.add_argument("--height", help="Resolution height.", default=480)
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
    FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP)

    # Initialize engine.
    engine = BasicEngine(args.model)

    is_inpaint_mode = False
    resolution_width = args.width
    rezolution_height = args.height
    with picamera.PiCamera() as camera:

        camera.resolution = (resolution_width, rezolution_height)
        camera.framerate = 30
        _, width, height, channels = engine.get_input_tensor_shape()
        rawCapture = PiRGBArray(camera)

        # allow the camera to warmup
        time.sleep(0.1)

        try:
            for frame in camera.capture_continuous(
                rawCapture, format="rgb", use_video_port=True
            ):
                start_ms = time.time()

                rawCapture.truncate(0)
                image = frame.array

                # Create inpute tensor
                # camera resolution (640, 480) => input tensor size (513, 513)
                input_buf = Image.fromarray(image)
                input_buf = input_buf.resize((width, height), Image.NEAREST)
                input_tensor = np.asarray(input_buf).flatten()

                # Run inference
                latency, result = engine.RunInference(input_tensor)

                # Create segmentation map
                seg_map = np.array(result, dtype=np.uint8)
                seg_map = np.reshape(seg_map, (width, height))
                seg_image = label_to_color_image(seg_map)
                # segmentation map resize 513, 513 => camera resolution(640, 480)
                seg_image = cv2.resize(seg_image, (resolution_width, rezolution_height))
                out_image = image // 2 + seg_image // 2
                im = cv2.cvtColor(out_image, cv2.COLOR_RGB2BGR)  # display image

                elapsed_ms = time.time() - start_ms

                # Calc fps.
                fps = 1 / elapsed_ms
                fps_text = "{0:.2f}ms, {1:.2f}fps".format((elapsed_ms * 1000.0), fps)
                visual.draw_caption(im, (10, 30), fps_text)

                latency_text = "Runinference latency: {0:.2f}ms".format(latency)
                visual.draw_caption(im, (10, 60), latency_text)

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
