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

import cv2
import numpy as np

from utils import visualization as visual
from utils.tflite_util import make_interpreter, set_input_tensor, get_output_tensor


WINDOW_NAME = "TF-lite ESRGAN (OpenCV)"


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
    parser.add_argument("--model", help="File path of TF-lite esrgan model.", required=True)
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
    _, esrgan_height, esrgan_width, esrgan_channel = input_details[0]["shape"]
    print(
        "ESRGAN interpreter(height, width, channel): ",
        esrgan_height,
        esrgan_width,
        esrgan_channel,
    )

    # Video capture.
    print("Open camera.")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    camera_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    camera_heigth = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("Camera Input(height, width, fps): ", camera_heigth, camera_width, fps)

    # target 
    is_super_resolution = False
    xmin = (camera_width // 2) - (esrgan_width // 2)
    xmax = xmin + esrgan_width
    ymin = (camera_heigth // 2) - (esrgan_width // 2)
    ymax = ymin + 50

    elapsed_list = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("VideoCapture read return false.")
            break

        if is_super_resolution:
            im = frame[ymin: ymax, xmin : xmax]
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im = im[np.newaxis,:,:,:]
            im = im.astype(np.float32)

            interpreter.set_tensor(input_details[0]['index'], im.copy())
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])

            sr_im = np.squeeze(output_data, axis=0)
            sr_im = np.clip(sr_im, 0, 255)
            sr_im = np.round(sr_im)
            sr_im = sr_im.astype(np.uint8)
            sr_im = cv2.cvtColor(sr_im, cv2.COLOR_RGB2BGR)

            print(sr_im.shape)

            output_height, output_width = sr_im.shape[:2]
            x = (camera_width // 2) - (output_width // 2)
            y = (camera_heigth // 2) - (output_height // 2)
            frame[y:y+output_height, x:x+output_width] = sr_im

        else:
            visual.draw_rectangle(frame, (xmin, ymin, xmax, ymax), (255, 255, 0))

        # Display
        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(10) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            is_super_resolution = not is_super_resolution

    # When everything done, release the window
    cap.release()

if __name__ == "__main__":
    main()
