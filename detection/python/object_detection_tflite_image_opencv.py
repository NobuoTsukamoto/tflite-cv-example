#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    TensorFlow Lite Object detection with OpenCV.

    Copyright (c) 2022 Nobuo Tsukamoto

    This software is released under the MIT License.
    See the LICENSE file in the project root for more information.
"""

import argparse
import random

import cv2
import numpy as np
from utils import visualization as visual
from utils.label_util import read_label_file
from utils.tflite_util import get_output_tensor, make_interpreter, set_input_tensor

WINDOW_NAME = "TF-lite object detection (OpenCV)"


def get_output(interpreter, score_threshold):
    """Returns list of detected objects.

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
    parser.add_argument("--label", help="File path of label file.", required=True)
    parser.add_argument("--image", help="File path of input image file.", required=True)
    parser.add_argument(
        "--output", help="File path of result image file.", required=True
    )
    parser.add_argument(
        "--threshold", help="threshold to filter results.", default=0.5, type=float
    )
    parser.add_argument("--width", help="Resolution width.", default=640, type=int)
    parser.add_argument("--height", help="Resolution height.", default=480, type=int)
    parser.add_argument("--thread", help="Num threads.", default=2, type=int)
    parser.add_argument("--delegate", help="File path of result.", default=None)
    args = parser.parse_args()

    # Initialize TF-Lite interpreter.
    interpreter = make_interpreter(args.model, args.thread, args.delegate)
    interpreter.allocate_tensors()
    _, height, width, channel = interpreter.get_input_details()[0]["shape"]
    print("Interpreter(height, width, channel): ", height, width, channel)

    # Read label and generate random colors.
    labels = read_label_file(args.label) if args.label else None
    last_key = sorted(labels.keys())[len(labels.keys()) - 1]
    random.seed(42)
    colors = visual.random_colors(last_key)

    im = cv2.imread(args.image)

    h, w, _ = im.shape
    print("Input(height, width): ", h, w)

    input_im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    resize_im = cv2.resize(input_im, (args.width, args.height))
    # resize_im = resize_im / 127.5 -1.

    set_input_tensor(interpreter, resize_im)
    interpreter.invoke()
    objs = get_output(interpreter, args.threshold)

    # Display result.
    for i, obj in enumerate(objs):
        class_id = int(obj["class_id"])
        caption = "{0}({1:.2f})".format(labels[class_id], obj["score"])

        # Convert the bounding box figures from relative coordinates
        # to absolute coordinates based on the original resolution
        ymin, xmin, ymax, xmax = obj["bounding_box"]
        xmin = int(xmin * w)
        xmax = int(xmax * w)
        ymin = int(ymin * h)
        ymax = int(ymax * h)

        # Draw a rectangle and caption.
        visual.draw_rectangle(im, (xmin, ymin, xmax, ymax), colors[class_id])
        visual.draw_caption(im, (xmin, ymin, xmax, ymax), caption)

        # Print output
        print(
            "{}: {}, {}, ({}, {}), ({}, {})".format(
                i, labels[class_id], obj["score"], xmin, ymin, xmax, ymax
            )
        )

    # Output image.
    cv2.imwrite(args.output, im)


if __name__ == "__main__":
    main()
