#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Edge TPU image classify Raspberry Pi camera stream.

    Copyright (c) 2019 Nobuo Tsukamoto

    This software is released under the MIT License.
    See the LICENSE file in the project root for more information.
"""
import argparse
import io
import time

import numpy as np
import picamera
from picamera.array import PiRGBArray

import edgetpu.classification.engine

import cv2
import PIL

sys.path.append(os.pardir)

WINDOW_NAME = 'Edge TPU Image classification'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
      '--model', help='File path of Tflite model.', required=True)
    parser.add_argument(
      '--label', help='File path of label file.', required=True)
    parser.add_argument('--top_k', help=")
    args = parser.parse_args()

    with open(args.label, 'r') as f:
        pairs = (l.strip().split(maxsplit=1) for l in f.readlines())
        labels = dict((int(k), v) for k, v in pairs)


    # Initialize window.
    cv2.namedWindow(WINDOW_NAME)
    cv2.moveWindow(WINDOW_NAME, 100, 200)

    # Initialize engine.
    engine = edgetpu.classification.engine.ClassificationEngine(args.model)

    width = 640
    height = 480
    elapsed_list = []
    with picamera.PiCamera() as camera:
        camera.resolution = (width, height)
        camera.framerate = 30
        # _, width, height, channels = engine.get_input_tensor_shape()

        rawCapture = PiRGBArray(camera)

        # allow the camera to warmup
        time.sleep(0.1)
        try:
            for frame in camera.capture_continuous(rawCapture,
                                                 format='rgb',
                                                 use_video_port=True):
                rawCapture.truncate(0)

                image = frame.array
                im = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                input_buf = PIL.Image.fromarray(image)

                start_ms = time.time()
                results = engine.ClassifyWithImage(input_buf, top_k=3)
                elapsed_ms = time.time() - start_ms

                # Check result.
                if results:
                    for i in range(len(results)):
                        label = '{0} ({1:.2f})'.format(
                            labels[results[i][0]], results[i][1])
                        pos = 60 + (i * 30)
                        draw_caption(im, (10, pos), label)

                # Calc fps.
                fps = 1 / elapsed_ms
                elapsed_list.append(elapsed_ms)
                avg_text = ''
                if len(elapsed_list) > 100:
                    elapsed_list.pop(0)
                    avg_elapsed_ms = np.mean(elapsed_list)
                    avg_fps = 1 / avg_elapsed_ms
                    avg_text = ' AGV: {0:.2f}ms, {1:.2f}fps'.format(
                        (avg_elapsed_ms * 1000.0), avg_fps)

                # Display fps
                fps_text = '{0:.2f}ms, {1:.2f}fps'.format(
                        (elapsed_ms * 1000.0), fps)
                draw_caption(im, (10, 30), fps_text + avg_text)
                
                 # display
                cv2.imshow(WINDOW_NAME, im)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

        finally:
            camera.stop_preview()


if __name__ == '__main__':
    main()
