#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Edge TPU object detection with OpenCV.

    Copyright (c) 2020 Nobuo Tsukamoto

    This software is released under the MIT License.
    See the LICENSE file in the project root for more information.
"""

import argparse
import time

import cv2
import numpy as np
from pycoral.adapters import common, detect
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from utils import visualization as visual

WINDOW_NAME = "Edge TPU PyCoral object detection (OpenCV)"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="File path of Tflite model.", required=True)
    parser.add_argument("--label", help="File path of label file.", required=True)
    parser.add_argument(
        "--threshold", help="threshold to filter results.", default=0.5, type=float
    )
    parser.add_argument("--width", help="Resolution width.", default=640, type=int)
    parser.add_argument("--height", help="Resolution height.", default=480, type=int)
    parser.add_argument("--videopath", help="File path of Videofile.", default="")
    args = parser.parse_args()

    # Initialize window.
    cv2.namedWindow(
        WINDOW_NAME, cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE | cv2.WINDOW_KEEPRATIO
    )
    cv2.moveWindow(WINDOW_NAME, 100, 200)

    # Initialize engine and load labels.
    interpreter = make_interpreter(args.model)
    interpreter.allocate_tensors()
    labels = read_label_file(args.label) if args.label else None

    # Generate random colors.
    last_key = sorted(labels.keys())[len(labels.keys()) - 1]
    colors = visual.random_colors(last_key)

    # Video capture.
    if args.videopath == "":
        print("Open camera.")
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    else:
        print("Open video file: ", args.videopath)
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

        elapsed_ms = (time.perf_counter() - start) * 1000

        # Display result.
        objects = detect.get_objects(interpreter, args.threshold, scale)
        if objects:
            for obj in objects:
                label_name = "Unknown"
                if labels:
                    labels.get(obj.id, "Unknown")
                    label_name = labels[obj.id]
                caption = "{0}({1:.2f})".format(label_name, obj.score)

                # Draw a rectangle and caption.
                box = (obj.bbox.xmin, obj.bbox.ymin, obj.bbox.xmax, obj.bbox.ymax)
                visual.draw_rectangle(frame, box, colors[obj.id])
                visual.draw_caption(frame, box, caption)

        # Calc fps.
        elapsed_list.append(elapsed_ms)
        avg_text = ""
        if len(elapsed_list) > 100:
            elapsed_list.pop(0)
            avg_elapsed_ms = np.mean(elapsed_list)
            avg_text = " AGV: {0:.2f}ms".format(avg_elapsed_ms)

        # Display fps
        fps_text = "{0:.2f}ms".format(elapsed_ms)
        visual.draw_caption(frame, (10, 30), fps_text + avg_text)

        # display
        cv2.imshow(WINDOW_NAME, frame)
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

    # When everything done, release the window
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
