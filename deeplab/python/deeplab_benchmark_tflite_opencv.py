#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Edge TPU DeepLabv3 Image segmenation benchmark with OpenCV.

    Copyright (c) 2020 Nobuo Tsukamoto

    This software is released under the MIT License.
    See the LICENSE file in the project root for more information.
"""
import argparse
import io
import os
import platform
import time

import cv2
import numpy as np
from utils.tflite_util import make_interpreter, set_input_tensor, get_output_tensor


def get_output(interpreter):
    """ Returns list of detected objects.
    """
    # Get all output details
    seg_map = get_output_tensor(interpreter, 0)
    return seg_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="File path of Tflite model.", required=True)
    parser.add_argument("--image", help="File path of image file.", required=True)
    parser.add_argument("--count", help="Repeat count.", default=100, type=int)
    parser.add_argument("--thread", help="Num threads.", default=2, type=int)
    args = parser.parse_args()

    # Initialize TF-Lite interpreter.
    interpreter = make_interpreter(args.model, args.thread)
    interpreter.allocate_tensors()
    _, height, width, channel = interpreter.get_input_details()[0]["shape"]
    print("Interpreter: ", height, width, channel)

    # Load image.
    im = cv2.imread(args.image)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    resize_im = cv2.resize(im, (width, height))

    model_file, *device = args.model.split("@")
    model_name = os.path.splitext(os.path.basename(model_file))[0]

    elapsed_list = []

    for i in range(args.count + 1):

        # Run inference.
        start = time.perf_counter()

        set_input_tensor(interpreter, resize_im)
        interpreter.invoke()
        seg_map = get_output(interpreter)

        inference_time = (time.perf_counter() - start) * 1000
        if i is 0:
            print("First Inference : {0:.2f} ms".format(inference_time))
        else:
            elapsed_list.append(inference_time)

    print("Inference : {0:.2f} ms".format(np.array(elapsed_list).mean()))


if __name__ == "__main__":
    main()
