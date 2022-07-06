#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Edge TPU image classification benchmark with OpenCV.

    Copyright (c) 2020 Nobuo Tsukamoto

    This software is released under the MIT License.
    See the LICENSE file in the project root for more information.
"""

import argparse
import time

import cv2
import numpy as np
from utils.tflite_util import make_interpreter, set_input_tensor


def load_labels(filename):
    with open(filename, "r") as f:
        return [line.strip() for line in f.readlines()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="File path of Tflite model.", required=True)
    parser.add_argument("--image", help="File path of image file.", required=True)
    parser.add_argument("--label", help="File path of label file.", required=True)
    parser.add_argument("--thread", help="Num threads.", default=2, type=int)
    parser.add_argument("--count", help="Repeat count.", default=100, type=int)
    parser.add_argument("--top_k", help="keep top k candidates.", default=3, type=int)
    args = parser.parse_args()

    # Read label file.
    labels = load_labels(args.label)

    # Initialize TF-Lite interpreter.
    interpreter = make_interpreter(args.model, args.thread)
    interpreter.allocate_tensors()
    _, height, width, channel = interpreter.get_input_details()[0]["shape"]
    output_details = interpreter.get_output_details()
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
        inference_time = (time.perf_counter() - start) * 1000

        output_data = interpreter.get_tensor(output_details[0]["index"])

        if i == 0:
            print("First Inference : {0:.2f} ms".format(inference_time))
        else:
            elapsed_list.append(inference_time)

    print("Inference : {0:.2f} ms".format(np.array(elapsed_list).mean()))

    results = np.squeeze(output_data)
    top_k = results.argsort()[-args.top_k:][::-1]
    # labels = load_labels(args.label_file)
    print(top_k)

    for index, result in enumerate(top_k):
        print(
            "{}: {}, score={:02f}".format(
                index, labels[result], results[result] / 255.0
            )
        )


if __name__ == "__main__":
    main()
