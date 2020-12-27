#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Edge TPU image classify with OpenCV.

    Copyright (c) 2020 Nobuo Tsukamoto

    This software is released under the MIT License.
    See the LICENSE file in the project root for more information.
"""
import argparse
import time

import cv2
import numpy as np
from pycoral.adapters import classify, common
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from utils import visualization as visual

WINDOW_NAME = "Edge TPU Image classification"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="File path of Tflite model.", required=True)
    parser.add_argument("--label", help="File path of label file.", required=True)
    parser.add_argument("--top_k", help="keep top k candidates.", default=3, type=int)
    parser.add_argument("--threshold", help="Score threshold.", default=0.0, type=float)
    parser.add_argument("--width", help="Resolution width.", default=640, type=int)
    parser.add_argument("--height", help="Resolution height.", default=480, type=int)
    parser.add_argument("--videopath", help="File path of Videofile.", default="")
    args = parser.parse_args()

    # Initialize window.
    cv2.namedWindow(WINDOW_NAME)
    cv2.moveWindow(WINDOW_NAME, 100, 200)

    # Initialize engine and load labels.
    interpreter = make_interpreter(args.model)
    interpreter.allocate_tensors()
    labels = read_label_file(args.label) if args.label else None

    # Video capture.
    if args.videopath == "":
        print("open camera.")
        cap = cv2.VideoCapture(-1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    else:
        print(args.videopath)
        cap = cv2.VideoCapture(args.videopath)

    cap_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    elapsed_list = []

    while cap.isOpened():
        _, frame = cap.read()
        im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run inference.
        start = time.perf_counter()

        _, scale = common.set_resized_input(
            interpreter, (cap_width, cap_height), lambda size: cv2.resize(im, size)
        )
        interpreter.invoke()

        # Check result.
        results = classify.get_classes(interpreter, args.top_k, args.threshold)
        elapsed_ms = (time.perf_counter() - start) * 1000
        if results:
            for i in range(len(results)):
                label = "{0} ({1:.2f})".format(labels[results[i][0]], results[i][1])
                pos = 60 + (i * 30)
                visual.draw_caption(frame, (10, pos), label)

        # Calc fps.
        fps = 1 / elapsed_ms * 1000
        elapsed_list.append(elapsed_ms)
        avg_text = ""
        if len(elapsed_list) > 100:
            elapsed_list.pop(0)
            avg_elapsed_ms = np.mean(elapsed_list)
            avg_fps = 1 / avg_elapsed_ms
            avg_text = " AGV: {0:.2f}ms, {1:.2f}fps".format(avg_elapsed_ms, avg_fps)

        # Display fps
        fps_text = "{0:.2f}ms, {1:.2f}fps".format(elapsed_ms, fps)
        visual.draw_caption(frame, (10, 30), fps_text + avg_text)

        # display
        cv2.imshow(WINDOW_NAME, frame)
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break


if __name__ == "__main__":
    main()
