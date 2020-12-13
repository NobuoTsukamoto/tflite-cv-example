#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    TensorFlow Lite Super resolution(ESRGAN) Capture with OpenCV.

    Copyright (c) 2020 Nobuo Tsukamoto

    This software is released under the MIT License.
    See the LICENSE file in the project root for more information.
"""

import argparse
import os
import random
import time
from itertools import cycle

import cv2
import numpy as np

from utils.tflite_util import get_output_tensor, make_interpreter, set_input_tensor

WINDOW_NAME = "TF-lite ESRGAN (OpenCV)"

WHITE = (240, 250, 250)

MODE = cycle(["normal", "esrgan", "resize"])


def drawTargetScope(
    frame, xmin, ymin, xmax, ymax, is_draw_text=True
):
    # Display Target Scorp
    points1 = np.array([(xmin, ymin + 15), (xmin, ymin), (xmin + 15, ymin)])
    cv2.polylines(frame, [points1], False, WHITE, thickness=2)
    points2 = np.array([(xmax - 15, ymin), (xmax, ymin), (xmax, ymin + 15)])
    cv2.polylines(frame, [points2], False, WHITE, thickness=2)
    points3 = np.array([(xmax, ymax - 15), (xmax, ymax), (xmax - 15, ymax)])
    cv2.polylines(frame, [points3], False, WHITE, thickness=2)
    points4 = np.array([(xmin, ymax - 15), (xmin, ymax), (xmin + 15, ymax)])
    cv2.polylines(frame, [points4], False, WHITE, thickness=2)
    if is_draw_text:
        points5 = np.array(
            [(xmax + 5, ymin - 5), (xmax + 25, ymin - 25), (xmax + 65, ymin - 25)]
        )
    else:
        points5 = np.array(
            [(xmax + 5, ymin - 5), (xmax + 25, ymin - 25), (xmax + 40, ymin - 25)]
        )
    cv2.polylines(frame, [points5], False, WHITE, thickness=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", help="File path of TF-lite esrgan model.", required=True
    )
    parser.add_argument("--width", help="Resolution width.", default=640, type=int)
    parser.add_argument("--height", help="Resolution height.", default=480, type=int)
    parser.add_argument("--thread", help="Num threads.", default=2, type=int)
    args = parser.parse_args()

    # Initialize window.
    cv2.namedWindow(
        WINDOW_NAME, cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE | cv2.WINDOW_KEEPRATIO
    )
    cv2.moveWindow(WINDOW_NAME, 100, 200)

    # Initialize TF-Lite interpreter.
    interpreter = make_interpreter(args.model, args.thread)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    _, input_height, input_width, input_channel = input_details[0]["shape"]
    _, output_height, output_width, output_channel = output_details[0]["shape"]
    print(
        "ESRGAN interpreter (%d, %d, %d) => (%d, %d, %d)."
        % (
            input_height,
            input_width,
            output_channel,
            output_height,
            output_width,
            output_channel,
        )
    )
    model_name = os.path.splitext(os.path.basename(args.model))[0]
    factor = output_height / input_width

    # Video capture.
    print("Open camera.")
    cap = cv2.VideoCapture(-1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    camera_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    camera_heigth = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("Camera Input(height, width, fps): ", camera_heigth, camera_width, fps)

    # target
    is_super_resolution = False
    xmin = (camera_width // 2) - (input_width // 2)
    xmax = xmin + input_width
    ymin = (camera_heigth // 2) - (input_height // 2)
    ymax = ymin + 50

    elapsed_list = []

    mode = next(MODE)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("VideoCapture read return false.")
            break

        if mode == "esrgan":
            # Create input
            im = frame[ymin:ymax, xmin:xmax]
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im = im[np.newaxis, :, :, :]
            im = im.astype(np.float32)

            # Run inference.
            start = time.perf_counter()

            interpreter.set_tensor(input_details[0]["index"], im.copy())
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]["index"])

            inference_time = (time.perf_counter() - start) * 1000

            # Display result.
            sr_im = np.squeeze(output_data, axis=0)
            sr_im = np.clip(sr_im, 0, 255)
            sr_im = np.round(sr_im)
            sr_im = sr_im.astype(np.uint8)
            sr_im = cv2.cvtColor(sr_im, cv2.COLOR_RGB2BGR)

            x = xmax + 40
            y = ymin + 10 - output_height
            frame[y : y + output_height, x : x + output_width] = sr_im

            points = np.array(
                [
                    (x, y),
                    (x + output_width, y),
                    (x + output_width, y + output_height),
                    (x, y + output_height),
                ]
            )
            cv2.polylines(frame, [points], True, WHITE, thickness=2)
            cv2.putText(
                frame,
                model_name
                + " (x"
                + str(factor)
                + ", {0:.2f}ms )".format(inference_time),
                (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                WHITE,
                1,
            )
            drawTargetScope(frame, xmin, ymin, xmax, ymax, False)

        elif mode == "resize":
            im = frame[ymin:ymax, xmin:xmax]
            resize_im = cv2.resize(
                im, (output_height, output_width), interpolation=cv2.INTER_CUBIC
            )

            x = xmax + 40
            y = ymin + 10 - output_height
            frame[y : y + output_height, x : x + output_width] = resize_im

            points = np.array(
                [
                    (x, y),
                    (x + output_width, y),
                    (x + output_width, y + output_height),
                    (x, y + output_height),
                ]
            )
            cv2.polylines(frame, [points], True, WHITE, thickness=2)
            cv2.putText(
                frame,
                " Bicubic (x" + str(factor) + ")",
                (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                WHITE,
                1,
            )
            drawTargetScope(frame, xmin, ymin, xmax, ymax, False)

        else:
            drawTargetScope(frame, xmin, ymin, xmax, ymax)
            cv2.putText(
                frame,
                "Target",
                (xmax + 25, ymin - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                WHITE,
                1,
            )

        # Display
        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(10) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            mode = next(MODE)
            print("Now Mode: ", mode)

    # When everything done, release the window
    cap.release()


if __name__ == "__main__":
    main()
