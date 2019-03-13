#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Edge TPU object detection Raspberry Pi camera stream.

    Copyright (c) 2019 Nobuo Tsukamoto

    This software is released under the MIT License.
    See the LICENSE file in the project root for more information.
"""

import argparse
import io
import time
import colorsys
import random

import numpy as np
import picamera
from picamera.array import PiRGBArray

from edgetpu.detection.engine import DetectionEngine

import cv2
import PIL

WINDOW_NAME = 'Edge TPU Tf-lite object detection'

# Function to read labels from text files.
def ReadLabelFile(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    ret = {}
    for line in lines:
        pair = line.strip().split(maxsplit=1)
        ret[int(pair[0])] = pair[1].strip()
    return ret

def random_colors(N):
    """ Random color generator.
    """
    hsv = [(i / N, 1.0, 1.0) for i in range(N)]
    colors = list(map(lambda c: tuple(int(i * 255) for i in colorsys.hsv_to_rgb(*c)), hsv))
    random.shuffle(colors)
    return colors

def draw_rectangle(image, box, color, thickness=1):
    """ Draws a rectangle.

    Args:
        image: The image to draw on.
        box: A list of 4 elements (x1, y1, x2, y2).
        color: Rectangle color.
        thickness: Thickness of lines.
    """
    b = np.array(box).astype(int)
    cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), color)

def draw_caption(image, box, caption):
    """ Draws a caption above the box in an image.

    Args:
        image: The image to draw on.
        box: A list of 4 elements (x1, y1, x2, y2).
        caption: String containing the text to draw.
    """
    b = np.array(box).astype(int)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
      '--model', help='File path of Tflite model.', required=True)
    parser.add_argument(
      '--label', help='File path of label file.', required=True)
    args = parser.parse_args()

    # Initialize window.
    cv2.namedWindow(WINDOW_NAME)
    cv2.moveWindow(WINDOW_NAME, 100, 200)

    # Initialize engine.
    engine = DetectionEngine(args.model)
    labels = ReadLabelFile(args.label) if args.label else None

    # Generate random colors.
    last_key = sorted(labels.keys())[len(labels.keys()) - 1]
    colors = random_colors(last_key)

    elapsed_list = []
    with picamera.PiCamera() as camera:
        resolution_width = 640
        rezolution_height = 480

        camera.resolution = (resolution_width, rezolution_height)
        camera.framerate = 30
        _, width, height, channels = engine.get_input_tensor_shape()
        rawCapture = PiRGBArray(camera)

        # allow the camera to warmup
        time.sleep(0.1)

        try:
            for frame in camera.capture_continuous(rawCapture,
                                                 format='rgb',
                                                 use_video_port=True):
                rawCapture.truncate(0)

                # input_buf = np.frombuffer(stream.getvalue(), dtype=np.uint8)
                image = frame.array
                im = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                input_buf = PIL.Image.fromarray(image)

                # Run inference.
                start_ms = time.time()
                ans = engine.DetectWithImage(input_buf, threshold=0.5,
                       keep_aspect_ratio=False, relative_coord=False, top_k=10)
                # ans = engine.DetectWithInputTensor(input_buf, threshold=0.05,
                #         keep_aspect_ratio=False, relative_coord=False, top_k=10)
                elapsed_ms = time.time() - start_ms


                # Display result.
                if ans:
                    for obj in ans:
                        label_name = 'Unknown'
                        if labels:
                            label_name = labels[obj.label_id]
                        caption = '{0}({1:.3f})'.format(label_name, obj.score)

                        # Draw a rectangle and caption.
                        box = obj.bounding_box.flatten().tolist()
                        draw_rectangle(im, box, colors[obj.label_id])
                        draw_caption(im, box, caption)

                # Calc fps.
                fps = 1 / elapsed_ms
                elapsed_list.append(elapsed_ms)
                avg_text = ""
                if len(elapsed_list) > 100:
                    elapsed_list.pop(0)
                    avg_elapsed_ms = np.mean(elapsed_list)
                    avg_fps = 1 / avg_elapsed_ms
                    avg_text = ' AGV: {0:.2f}ms, {1:.2f}fps'.format(
                        (avg_elapsed_ms * 1000.0), avg_fps)

                # Display fps
                fps_text = '{0:.2f}ms, {1:.2f}fps'.format(
                        (elapsed_ms * 1000.0), fps)
                draw_caption(im, (10, 50), fps_text + avg_text)

                # display
                cv2.imshow(WINDOW_NAME, im)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

        finally:
            camera.stop_preview()

    # When everything done, release the window
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
