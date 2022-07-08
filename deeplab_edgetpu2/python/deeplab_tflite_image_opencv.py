#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    DeepLab V3+ EdgeTPUV2 and AutoSeg EdgeTPU Image segmenation with OpenCV.

    Copyright (c) 2022 Nobuo Tsukamoto

    This software is released under the MIT License.
    See the LICENSE file in the project root for more information.
"""
import argparse

import cv2
import numpy as np
from utils import label_util
from utils.tflite_util import get_output_tensor, make_interpreter, set_input_tensor


def normalize(im):
    im = np.expand_dims(im, axis=0)
    im = (im - 128).astype(np.int8)
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

    # Initialize colormap
    colormap = label_util.create_pascal_label_colormap()

    frame = cv2.imread(args.input)
    h, w, _ = frame.shape
    im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized_im = cv2.resize(im, (width, height))
    normalized_im = normalize(resized_im)

    # Run inference.
    set_input_tensor(interpreter, normalized_im)
    interpreter.invoke()
    seg_map = get_output(interpreter)

    # Display result
    seg_map = np.reshape(seg_map, (width, height)).astype(np.uint8)
    print(np.unique(seg_map))
    seg_image = label_util.label_to_color_image(colormap, seg_map)
    seg_image = cv2.resize(seg_image, (w, h))
    im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) // 2 + seg_image // 2
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)

    # Output image file.
    cv2.imwrite(args.output, im)


if __name__ == "__main__":
    main()
