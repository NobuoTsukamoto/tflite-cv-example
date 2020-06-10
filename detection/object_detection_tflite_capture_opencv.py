#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Edge TPU object detection with OpenCV.

    Copyright (c) 2020 Nobuo Tsukamoto

    This software is released under the MIT License.
    See the LICENSE file in the project root for more information.
"""

import os
import argparse
import time
import random
import collections

import numpy as np

import tflite_runtime.interpreter as tflite
import platform

import cv2

from utils import visualization as visual

WINDOW_NAME = "TF-lite object detection (OpenCV)"

EDGETPU_SHARED_LIB = {
  'Linux': 'libedgetpu.so.1',
  'Darwin': 'libedgetpu.1.dylib',
  'Windows': 'edgetpu.dll'
}[platform.system()]


def read_label_file(file_path):
    """ Function to read labels from text files.

    Args:
        file_path: File path to labels.
    """
    with open(file_path, "r") as f:
        lines = f.readlines()

    ret = {}
    for line in lines:
        pair = line.strip().split(maxsplit=1)
        ret[int(pair[0])] = pair[1].strip()
    return ret


def make_interpreter(model_file):
    model_name = os.path.splitext(os.path.basename(model_file))[0]
    model_file, *device = model_file.split('@')

    if 'edgetpu.tflite' in model_file:
        print('edgetpu')
        return tflite.Interpreter(
            model_path=model_file,
            experimental_delegates = [
                tflite.load_delegate(EDGETPU_SHARED_LIB,
                                    {'device': device[0]} if device else {})
                ])
    else:
        return tflite.Interpreter(
            model_path=model_file)
        



def set_input_tensor(interpreter, image):
    """ Sets the input tensor.

        Args:
            interpreter: Interpreter object.
            image: a function that takes a (width, height) tuple, and returns an RGB image resized to those dimensions.
    """
    tensor_index = interpreter.get_input_details()[0]['index']
    input_tensor = interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = image.copy()
    

def get_output_tensor(interpreter, index):
    """Returns the output tensor at the given index."""
    output_details = interpreter.get_output_details()[index]
    tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
    return tensor


def get_output(interpreter, score_threshold):
    """ Returns list of detected objects.
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
                'bounding_box': boxes[i],
                'class_id': class_ids[i],
                'score': scores[i]
            }
            results.append(result)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="File path of Tflite model.", required=True)
    parser.add_argument("--label", help="File path of label file.", required=True)
    parser.add_argument(
        "--threshold", help="threshold to filter results.", default=0.5, type=float
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
    interpreter = make_interpreter(args.model)
    interpreter.allocate_tensors()
    interpreter.set_num_threads(args.thread)
    _, height, width, channel = interpreter.get_input_details()[0]['shape']
    print('Interpreter: ', height, width, channel)

    # Read label and generate random colors.
    labels = read_label_file(args.label) if args.label else None
    last_key = sorted(labels.keys())[len(labels.keys()) - 1]
    random.seed(42)
    colors = visual.random_colors(last_key)

    # Video capture.
    if args.videopath == '':
        print('open camera.')
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    else:
        print(args.videopath)
        cap = cv2.VideoCapture(args.videopath)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print('Input: ', h, w, fps)

    model_file, *device = args.model.split('@')
    model_name = os.path.splitext(os.path.basename(model_file))[0]

    # Output Video file
    # Define the codec and create VideoWriter object
    video_writer = None
    if args.output != '' :
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        video_writer = cv2.VideoWriter(args.output, fourcc, fps, (w, h))

    elapsed_list = []

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            print('VideoCapture read return false.')
            break

        im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resize_im = cv2.resize(im, (width, height))

        # Run inference.
        start = time.perf_counter()

        set_input_tensor(interpreter, resize_im)
        interpreter.invoke()
        objs = get_output(interpreter, args.threshold)

        inference_time = (time.perf_counter() - start) * 1000

        # Display result.
        for obj in objs:
            class_id = int(obj['class_id'])
            caption = "{0}({1:.2f})".format(labels[class_id], obj['score'])

            # Convert the bounding box figures from relative coordinates
            # to absolute coordinates based on the original resolution
            ymin, xmin, ymax, xmax = obj['bounding_box']
            xmin = int(xmin * w)
            xmax = int(xmax * w)
            ymin = int(ymin * h)
            ymax = int(ymax * h)

            # Draw a rectangle and caption.
            visual.draw_rectangle(frame, (xmin, ymin, xmax, ymax), colors[class_id])
            visual.draw_caption(frame, (xmin, ymin, xmax, ymax), caption)

        # Calc fps.
        elapsed_list.append(inference_time)
        avg_text = ""
        if len(elapsed_list) > 100:
            elapsed_list.pop(0)
            avg_elapsed_ms = np.mean(elapsed_list)
            avg_text = " AGV: {0:.2f}ms".format(avg_elapsed_ms)

        # Display fps
        fps_text = "Inference: {0:.2f}ms".format(inference_time)
        display_text = model_name + ' ' + fps_text + avg_text
        visual.draw_caption(frame, (10, 30), display_text)

        # Output video file
        if video_writer != None:
            video_writer.write(frame)

        # Display
        cv2.imshow(WINDOW_NAME, frame)
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break


    # When everything done, release the window
    cap.release()
    if video_writer != None:
        video_writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
