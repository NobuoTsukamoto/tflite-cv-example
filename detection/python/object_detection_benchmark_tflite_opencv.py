#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    TensorFlow Lite Object detection benchmark with OpenCV.

    Copyright (c) 2020 Nobuo Tsukamoto

    This software is released under the MIT License.
    See the LICENSE file in the project root for more information.
"""

import argparse
import time

import cv2
import numpy as np
from utils.tflite_util import (get_output_tensor, make_interpreter,
                               set_input_tensor)


def get_output(interpreter, score_threshold):
    """ Returns list of detected objects.

    Args:
        interpreter
        score_threshold

    Returns: bounding_box, class_id, score
    """
    # Get all output details
    boxes = get_output_tensor(interpreter, 0)
    class_ids = get_output_tensor(interpreter, 1)
    scores = get_output_tensor(interpreter, 2)
    count = int(get_output_tensor(interpreter, 3))

    results = []
    for i in range(count):
        if scores[i] >= score_threshold:
            result = {
                "bounding_box": boxes[i],
                "class_id": class_ids[i],
                "score": scores[i],
            }
            results.append(result)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="File path of Tflite model.", required=True)
    parser.add_argument("--image", help="File path of image file.", required=True)
    parser.add_argument("--thread", help="Num threads.", default=2, type=int)
    parser.add_argument("--count", help="Repeat count.", default=100, type=int)
    parser.add_argument(
        "--threshold", help="threshold to filter results.", default=0.5, type=float
    )
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
        get_output(interpreter, args.threshold)

        inference_time = (time.perf_counter() - start) * 1000

        if i == 0:
            print("First Inference : {0:.2f} ms".format(inference_time))
        else:
            elapsed_list.append(inference_time)

    print("Inference : {0:.2f} ms".format(np.array(elapsed_list).mean()))


if __name__ == "__main__":
    main()
