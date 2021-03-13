#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    TensorFlow Lite CenterNet with OpenCV.

    Copyright (c) 2021 Nobuo Tsukamoto

    This software is released under the MIT License.
    See the LICENSE file in the project root for more information.
"""

import argparse
import os
import random
import time

import cv2
import numpy as np
from utils import visualization as visual
from utils.label_util import read_label_file
from utils.tflite_util import get_output_tensor, make_interpreter, set_input_tensor

WINDOW_NAME = "TF-lite object detection (OpenCV)"

KEYPOINT_EDGES = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (0, 5),
    (0, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]


def get_output(interpreter, score_threshold, is_keypoints=False):
    """Returns list of detected objects.

    Args:
        interpreter
        score_threshold
        is_keypoints

    Returns: bounding_box, class_id, score
    """
    # Get all output details
    boxes = get_output_tensor(interpreter, 0)
    class_ids = get_output_tensor(interpreter, 1)
    scores = get_output_tensor(interpreter, 2)
    count = int(get_output_tensor(interpreter, 3))
    keypoints = None
    keypoints_scores = None
    if is_keypoints:
        keypoints = get_output_tensor(interpreter, 4)
        keypoints_scores = get_output_tensor(interpreter, 5)

    results = []
    for i in range(count):
        if scores[i] >= score_threshold:
            result = {
                "bounding_box": boxes[i],
                "class_id": class_ids[i],
                "box_score": scores[i],
            }
            if is_keypoints:
                keypoint_result = {
                    "keypoints": keypoints[i],
                    "keypoints_score": keypoints_scores[i],
                }
                result.update(keypoint_result)
            results.append(result)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="File path of Tflite model.", required=True)
    parser.add_argument("--label", help="File path of label file.", required=True)
    parser.add_argument(
        "--threshold", help="threshold to filter results.", default=0.5, type=float
    )
    parser.add_argument(
        "--keypoint", help="Include keypoint detection.", action="store_true"
    )
    parser.add_argument("--width", help="Resolution width.", default=640, type=int)
    parser.add_argument("--height", help="Resolution height.", default=480, type=int)
    parser.add_argument("--thread", help="Num threads.", default=2, type=int)
    parser.add_argument("--videopath", help="File path of Videofile.", default="")
    parser.add_argument("--output", help="File path of result.", default="")
    args = parser.parse_args()

    # Initialize window.
    cv2.namedWindow(
        WINDOW_NAME, cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE | cv2.WINDOW_KEEPRATIO
    )
    cv2.moveWindow(WINDOW_NAME, 100, 200)

    # Initialize TF-Lite interpreter.
    interpreter = make_interpreter(args.model, args.thread)
    interpreter.allocate_tensors()
    _, height, width, channel = interpreter.get_input_details()[0]["shape"]
    print("Interpreter(height, width, channel): ", height, width, channel)

    # Read label and generate random colors.
    labels = read_label_file(args.label) if args.label else None
    last_key = sorted(labels.keys())[len(labels.keys()) - 1]
    random.seed(42)
    colors = visual.random_colors(last_key)

    # Video capture.
    if args.videopath == "":
        print("open camera.")
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    else:
        print("open video file", args.videopath)
        cap = cv2.VideoCapture(args.videopath)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("Input(height, width, fps): ", h, w, fps)

    model_name = os.path.splitext(os.path.basename(args.model))[0]

    # Output Video file
    # Define the codec and create VideoWriter object
    video_writer = None
    if args.output != "":
        fourcc = cv2.VideoWriter_fourcc(*"MP4V")
        video_writer = cv2.VideoWriter(args.output, fourcc, fps, (w, h))

    elapsed_list = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("VideoCapture read return false.")
            break

        im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resize_im = cv2.resize(im, (width, height))

        # Run inference.
        start = time.perf_counter()

        set_input_tensor(interpreter, resize_im)
        interpreter.invoke()
        objs = get_output(interpreter, args.threshold, args.keypoint)


        # Display result.
        for obj in objs:
            class_id = int(obj["class_id"])
            caption = "{0}({1:.2f})".format(labels[class_id], obj["box_score"])

            # Convert the bounding box figures from relative coordinates
            # to absolute coordinates based on the original resolution
            ymin, xmin, ymax, xmax = obj["bounding_box"]
            xmin = int(xmin * w)
            xmax = int(xmax * w)
            ymin = int(ymin * h)
            ymax = int(ymax * h)

            # Draw a rectangle and caption.
            visual.draw_rectangle(frame, (xmin, ymin, xmax, ymax), colors[class_id])
            visual.draw_caption(frame, (xmin, ymin, xmax, ymax), caption)

            # Draw keypoint
            if args.keypoint:
                keypoints = np.array(obj["keypoints"])
                keypoints_x = [int(i[1] * w) for i in keypoints]
                keypoints_y = [int(i[0] * h) for i in keypoints]
                keypoint_scores = np.array(obj["keypoints_score"])
                valid_keypoint = np.greater(keypoint_scores, args.threshold)
                valid_keypoint = [j for j in valid_keypoint]
                for keypoint_x, keypoint_y, valid in zip(
                    keypoints_x, keypoints_y, valid_keypoint
                ):
                    if valid:
                        visual.draw_circle(frame, (keypoint_x, keypoint_y))

                for keypoint_start, keypoint_end in KEYPOINT_EDGES:
                    if (
                        keypoint_start < 0
                        or keypoint_start >= len(keypoints)
                        or keypoint_end < 0
                        or keypoint_end >= len(keypoints)
                    ):
                        continue
                    if not (
                        valid_keypoint[keypoint_start] and valid_keypoint[keypoint_end]
                    ):
                        continue
                    visual.draw_line(
                        frame,
                        (keypoints_x[keypoint_start], keypoints_y[keypoint_start]),
                        (keypoints_x[keypoint_end], keypoints_y[keypoint_end]),
                    )

        inference_time = (time.perf_counter() - start) * 1000

        # Calc fps.
        elapsed_list.append(inference_time)
        avg_text = ""
        if len(elapsed_list) > 100:
            elapsed_list.pop(0)
            avg_elapsed_ms = np.mean(elapsed_list)
            avg_text = " AGV: {0:.2f}ms".format(avg_elapsed_ms)

        # Display fps
        fps_text = "Inference: {0:.2f}ms".format(inference_time)
        display_text = model_name + " " + fps_text + avg_text
        visual.draw_caption(frame, (10, 30), display_text)

        # Output video file
        if video_writer is not None:
            video_writer.write(frame)

        # Display
        cv2.imshow(WINDOW_NAME, frame)
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

    # When everything done, release the window
    cap.release()
    if video_writer is not None:
        video_writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
