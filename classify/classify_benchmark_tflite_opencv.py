#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Edge TPU image classification benchmark with OpenCV.

    Copyright (c) 2020 Nobuo Tsukamoto

    This software is released under the MIT License.
    See the LICENSE file in the project root for more information.
"""

import os
import argparse
import time
import collections
import operator

import numpy as np

import tflite_runtime.interpreter as tflite
import platform

import cv2

from utils.tflite_util import make_interpreter, set_input_tensor, get_output_tensor


Class = collections.namedtuple("Class", ["id", "score"])


def get_output(interpreter, top_k=1, score_threshold=0.0):
    """ Returns list of detected objects.
    """
    scores = get_output_tensor(interpreter, 0)
    classes = [
        Class(i, scores[i])
        for i in np.argpartition(scores, -top_k)[-top_k:]
        if scores[i] >= score_threshold
    ]
    return sorted(classes, key=operator.itemgetter(1), reverse=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="File path of Tflite model.", required=True)
    parser.add_argument("--image", help="File path of image file.", required=True)
    parser.add_argument("--thread", help="Num threads.", default=2, type=int)
    parser.add_argument("--count", help="Repeat count.", default=100, type=int)
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

    h, w = im.shape[:2]
    print("Input: ", h, w)

    elapsed_list = []

    for i in range(args.count + 1):
        # Run inference.
        start = time.perf_counter()

        set_input_tensor(interpreter, resize_im)
        interpreter.invoke()
        # objs = get_output(interpreter, args.threshold)
        classes = get_output(interpreter)

        inference_time = (time.perf_counter() - start) * 1000

        if i is 0:
            print("First Inference : {0:.2f} ms".format(inference_time))
        else:
            elapsed_list.append(inference_time)

    print("Inference : {0:.2f} ms".format(np.array(elapsed_list).mean()))


if __name__ == "__main__":
    main()
