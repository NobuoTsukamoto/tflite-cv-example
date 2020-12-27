#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    TF-Lite image segmentation with PiCamera.

    Copyright (c) 2020 Nobuo Tsukamoto

    This software is released under the MIT License.
    See the LICENSE file in the project root for more information.
"""

import os
import argparse
import time
import collections

import cv2
import numpy as np
import picamera
from picamera.array import PiRGBArray
from utils import visualization as visual
from utils.tflite_util import (get_output_tensor, make_interpreter,
                                       set_input_tensor)

WINDOW_NAME = "TF-Lite image segmentation (PiCamera)"


def create_mask(pred_mask):
    pred_mask = np.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., np.newaxis]
    return pred_mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="File path of Tflite model.", required=True)
    parser.add_argument(
        "--threshold", help="threshold to filter results.", default=0.5, type=float
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
    _, height, width, channel = interpreter.get_input_details()[0]["shape"]
    print("Interpreter: ", height, width, channel)

    model_name = os.path.splitext(os.path.basename(args.model))[0]

    elapsed_list = []

    resolution_width = args.width
    rezolution_height = args.height
    with picamera.PiCamera() as camera:

        camera.resolution = (resolution_width, rezolution_height)
        camera.framerate = 30
        # _, width, height, channels = engine.get_input_tensor_shape()
        rawCapture = PiRGBArray(camera)

        # allow the camera to warmup
        time.sleep(0.1)

        try:
            for frame in camera.capture_continuous(
                rawCapture, format="rgb", use_video_port=True
            ):
                rawCapture.truncate(0)

                start = time.perf_counter()

                image = frame.array
                im = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                resize_im = cv2.resize(im, (width, height))
                input_im = resize_im.astype(np.float32)
                input_im = input_im / 255

                # Run inference.
                set_input_tensor(interpreter, input_im[np.newaxis, :, :])
                interpreter.invoke()
                predictions = get_output_tensor(interpreter, 0)

                pred_mask = create_mask(predictions)
                pred_mask = np.array(pred_mask, dtype="uint8")
                pred_mask = pred_mask * 127
                pred_mask = cv2.resize(pred_mask, (resolution_width, rezolution_height))

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
                visual.draw_caption(im, (10, 30), display_text)

                # display
                pred_mask = cv2.cvtColor(pred_mask, cv2.COLOR_GRAY2BGR)
                display = cv2.hconcat([im, pred_mask])
                cv2.imshow(WINDOW_NAME, display)
                if cv2.waitKey(10) & 0xFF == ord("q"):
                    break

        finally:
            camera.stop_preview()

    # When everything done, release the window
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
