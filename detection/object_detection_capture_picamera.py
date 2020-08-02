#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Edge TPU object detection Raspberry Pi camera stream.

    Copyright (c) 2019 Nobuo Tsukamoto

    This software is released under the MIT License.
    See the LICENSE file in the project root for more information.
"""

import argparse
import time

import cv2
import numpy as np
import picamera
import PIL
from edgetpu.detection.engine import DetectionEngine
from picamera.array import PiRGBArray
from utils import visualization as visual
from utils.label_util import read_label_file

WINDOW_NAME = "Edge TPU TF-lite object detection(PiCamera)"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="File path of Tflite model.", required=True)
    parser.add_argument("--label", help="File path of label file.", required=True)
    parser.add_argument("--top_k", help="keep top k candidates.", default=3)
    parser.add_argument(
        "--threshold", help="threshold to filter results.", default=0.5, type=float
    )
    parser.add_argument("--width", help="Resolution width.", default=640, type=int)
    parser.add_argument("--height", help="Resolution height.", default=480, type=int)
    args = parser.parse_args()

    # Initialize window.
    cv2.namedWindow(
        WINDOW_NAME, cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE | cv2.WINDOW_KEEPRATIO
    )
    cv2.moveWindow(WINDOW_NAME, 100, 200)

    # Initialize engine.
    engine = DetectionEngine(args.model)
    labels = read_label_file(args.label) if args.label else None

    # Generate random colors.
    last_key = sorted(labels.keys())[len(labels.keys()) - 1]
    colors = visual.random_colors(last_key)

    elapsed_list = []
    resolution_width = args.width
    rezolution_height = args.height
    with picamera.PiCamera() as camera:

        camera.resolution = (resolution_width, rezolution_height)
        camera.framerate = 30
        _, width, height, channels = engine.get_input_tensor_shape()
        rawCapture = PiRGBArray(camera)

        # allow the camera to warmup
        time.sleep(0.1)

        try:
            for frame in camera.capture_continuous(
                rawCapture, format="rgb", use_video_port=True
            ):
                rawCapture.truncate(0)

                # input_buf = np.frombuffer(stream.getvalue(), dtype=np.uint8)
                image = frame.array
                im = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                input_buf = PIL.Image.fromarray(image)

                # Run inference.
                ans = engine.detect_with_image(
                    input_buf,
                    threshold=args.threshold,
                    keep_aspect_ratio=False,
                    relative_coord=False,
                    top_k=args.top_k,
                )
                elapsed_ms = engine.get_inference_time()

                # Display result.
                if ans:
                    for obj in ans:
                        label_name = "Unknown"
                        if labels:
                            label_name = labels[obj.label_id]
                        caption = "{0}({1:.2f})".format(label_name, obj.score)

                        # Draw a rectangle and caption.
                        box = obj.bounding_box.flatten().tolist()
                        visual.draw_rectangle(im, box, colors[obj.label_id])
                        visual.draw_caption(im, box, caption)

                # Calc fps.
                elapsed_list.append(elapsed_ms)
                avg_text = ""
                if len(elapsed_list) > 100:
                    elapsed_list.pop(0)
                    avg_elapsed_ms = np.mean(elapsed_list)
                    avg_text = " AGV: {0:.2f}ms".format(avg_elapsed_ms)

                # Display fps
                fps_text = "{0:.2f}ms".format(elapsed_ms)
                visual.draw_caption(im, (10, 30), fps_text + avg_text)

                # display
                cv2.imshow(WINDOW_NAME, im)
                if cv2.waitKey(10) & 0xFF == ord("q"):
                    break

        finally:
            camera.stop_preview()

    # When everything done, release the window
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
