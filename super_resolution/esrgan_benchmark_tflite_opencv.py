#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    TensorFlow Lite Super resolution(ESRGAN) benchmark with OpenCV.

    Copyright (c) 2020 Nobuo Tsukamoto

    This software is released under the MIT License.
    See the LICENSE file in the project root for more information.
"""

import argparse
import collections
import operator
import os
import platform
import time

import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
from utils.tflite_util import make_interpreter


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

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Load image.
    im = cv2.imread(args.image)
    h, w = im.shape[:2]
    print("Input: ", h, w)

    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = im[np.newaxis, :, :, :]
    im = im.astype(np.float32)
    print(im.shape, im.dtype)
    # im = tf.expand_dims(im, axis=0)
    # im = tf.cast(lr, tf.float32)

    elapsed_list = []

    for i in range(args.count + 1):
        # Run inference.
        start = time.perf_counter()

        interpreter.set_tensor(input_details[0]["index"], im.copy())
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]["index"])

        inference_time = (time.perf_counter() - start) * 1000

        if i is 0:
            print("First Inference : {0:.2f} ms".format(inference_time))

            sr_im = np.squeeze(output_data, axis=0)
            sr_im = np.clip(sr_im, 0, 255)
            sr_im = np.round(sr_im)
            sr_im = sr_im.astype(np.uint8)
            sr_im = cv2.cvtColor(sr_im, cv2.COLOR_RGB2BGR)

            cv2.imwrite("./sr.png", sr_im)

        else:
            elapsed_list.append(inference_time)

    print("Inference : {0:.2f} ms".format(np.array(elapsed_list).mean()))


if __name__ == "__main__":
    main()
