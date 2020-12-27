#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Edge TPU object detection and camouflage object Raspberry Pi camera stream.

    Copyright (c) 2019 Nobuo Tsukamoto

    This software is released under the MIT License.
    See the LICENSE file in the project root for more information.
"""

import argparse
import io
import time

import cv2
import numpy as np
import picamera
from picamera.array import PiRGBArray
from pycoral.adapters import common, detect
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from utils import visualization as visual

WINDOW_NAME = "Inpaint Pi Camera"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="File path of Tflite model.", required=True)
    parser.add_argument("--label", help="File path of label file.", required=True)
    parser.add_argument(
        "--threshold", help="threshold to filter results.", type=float, default=0.5
    )
    parser.add_argument("--width", help="Resolution width.", default=640)
    parser.add_argument("--height", help="Resolution height.", default=480)
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

    is_inpaint_mode = False
    resolution_width = args.width
    rezolution_height = args.height
    with picamera.PiCamera() as camera:

        camera.resolution = (resolution_width, rezolution_height)
        camera.framerate = 30
        rawCapture = PiRGBArray(camera)

        # allow the camera to warmup
        time.sleep(0.1)

        try:
            for frame in camera.capture_continuous(
                rawCapture, format="rgb", use_video_port=True
            ):
                start_ms = time.time()

                rawCapture.truncate(0)

                image = frame.array
                im = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # Run inference.
                start = time.perf_counter()

                _, scale = common.set_resized_input(
                    interpreter, (resolution_width, rezolution_height), lambda size: cv2.resize(image, size)
                )
                interpreter.invoke()

                # Display result.
                objects = detect.get_objects(interpreter, args.threshold, scale)

                if is_inpaint_mode == True:
                    mask = np.full((args.height, args.width), 0, dtype=np.uint8)
                    for obj in objects:
                        if labels and obj.id in labels:
                            # Draw a mask rectangle.
                            box = (obj.bbox.xmin, obj.bbox.ymin, obj.bbox.xmax, obj.bbox.ymax)
                            visual.draw_rectangle(
                                mask, box, (255, 255, 255), thickness=-1
                            )

                    # Image Inpainting
                    dst = cv2.inpaint(im, mask, 3, cv2.INPAINT_TELEA)
                    # dst = cv2.inpaint(im, mask,3,cv2.INPAINT_NS)

                else:
                    for obj in objects:
                        if labels and obj.id in labels:
                            label_name = labels[obj.id]
                            caption = "{0}({1:.2f})".format(label_name, obj.score)

                            # Draw a rectangle and caption.
                            box = (obj.bbox.xmin, obj.bbox.ymin, obj.bbox.xmax, obj.bbox.ymax)
                            visual.draw_rectangle(im, box, colors[obj.id])
                            visual.draw_caption(im, box, caption)
                    dst = im

                # Calc fps.
                elapsed_ms = time.time() - start_ms
                fps = 1 / elapsed_ms

                # Display fps
                fps_text = "{0:.2f}ms, {1:.2f}fps".format((elapsed_ms * 1000.0), fps)
                visual.draw_caption(dst, (10, 30), fps_text)

                # Display image
                cv2.imshow(WINDOW_NAME, dst)
                key = cv2.waitKey(10) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord(" "):
                    is_inpaint_mode = not is_inpaint_mode
                    print("inpant mode change :", is_inpaint_mode)

        finally:
            camera.stop_preview()

    # When everything done, release the window
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
