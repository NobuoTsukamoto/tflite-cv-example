#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    MOSAIC Image segmenation with OpenCV.

    Copyright (c) 2022 Nobuo Tsukamoto

    This software is released under the MIT License.
    See the LICENSE file in the project root for more information.
"""
import argparse
import os

import cv2
import numpy as np
from utils import label_util
from utils.tflite_util import get_output_tensor, make_interpreter, set_input_tensor

MEAN_RGB = [127.5, 127.5, 127.5]
STDDEV_RGB = [127.5, 127.5, 127.5]

COLORMAP = np.array(
    (
        (128, 64, 128),
        (244, 35, 232),
        (70, 70, 70),
        (102, 102, 156),
        (190, 153, 153),
        (153, 153, 153),
        (250, 170, 30),
        (220, 220, 0),
        (107, 142, 35),
        (152, 251, 152),
        (70, 130, 180),
        (220, 20, 60),
        (255, 0, 0),
        (0, 0, 142),
        (0, 0, 70),
        (0, 60, 100),
        (0, 80, 100),
        (0, 0, 230),
        (119, 11, 32),
        (0, 0, 0),
    ),
    np.uint8,
)


def normalize(im, quantize=True):
    if quantize:
        im = im.astype(np.int8)
    else:
        im = (im - MEAN_RGB) / STDDEV_RGB
    im = np.expand_dims(im, axis=0)
    return im


def get_output(interpreter):
    """Returns list of detected objects."""
    # Get all output details
    seg_map = get_output_tensor(interpreter, 0)
    return seg_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="File path of Tflite model.", required=True)
    parser.add_argument("--thread", help="Num threads.", default=2, type=int)
    parser.add_argument("--input", help="File path of image.", default="")
    parser.add_argument("--output", help="File path of result.", default="")
    args = parser.parse_args()

    # Initialize TF-Lite interpreter.
    interpreter = make_interpreter(args.model, args.thread)
    interpreter.allocate_tensors()
    _, height, width, channel = interpreter.get_input_details()[0]["shape"]
    print("Interpreter: ", height, width, channel)

    model_name = os.path.splitext(os.path.basename(args.model))[0]
    is_quantize = False
    if "quant" in model_name:
        is_quantize = True

    frame = cv2.imread(args.input)
    h, w, _ = frame.shape
    im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized_im = cv2.resize(im, (width, height))
    normalized_im = normalize(resized_im, is_quantize)

    # Run inference.
    set_input_tensor(interpreter, normalized_im)
    interpreter.invoke()
    seg_map = get_output(interpreter)

    # Display result
    seg_map = np.argmax(seg_map, axis=-1)
    seg_image = label_util.label_to_color_image(COLORMAP, seg_map.astype(np.uint8))
    seg_image = cv2.resize(seg_image, (w, h))
    im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) // 2 + seg_image // 2
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)

    # Output image file.
    cv2.imwrite(args.output, im)


if __name__ == "__main__":
    main()
