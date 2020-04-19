#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Edge TPU DeepLabv3 Image segmenation with OpenCV.

    Copyright (c) 2020 Nobuo Tsukamoto

    This software is released under the MIT License.
    See the LICENSE file in the project root for more information.
"""
import argparse
import io
import os
import random
import time

import numpy as np

import tflite_runtime.interpreter as tflite
import platform

import cv2

from utils import visualization as visual
from utils import label_util

WINDOW_NAME = 'Edge TPU Segmentation (OpenCV)'

def make_interpreter(model_file):
    model_name = os.path.splitext(os.path.basename(model_file))[0]
    model_file, *device = model_file.split('@')

    if 'edgetpu.tflite' in model_name:
        print('Use edgetpu')
        return tflite.Interpreter(
            model_path=model_file,
            experimental_delegates = [
                tflite.load_delegate(EDGETPU_SHARED_LIB,
                                    {'device': device[0]} if device else {})
                ])
    else:
        return tflite.Interpreter(model_path=model_file)


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
    

def get_output(interpreter):
    """ Returns list of detected objects.
    """
    # Get all output details
    seg_map = get_output_tensor(interpreter, 0)
    return seg_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='File path of Tflite model.', required=True)
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

    # Initialize colormap
    random.seed(42)
    colormap = label_util.create_pascal_label_colormap()

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
        seg_map = get_output(interpreter)

        inference_time = (time.perf_counter() - start) * 1000

        # Display result
        seg_map = np.reshape(seg_map, (width, height))
        seg_image = label_util.label_to_color_image(colormap, seg_map)
        seg_image = cv2.resize(seg_image, (w, h))
        im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) // 2 + seg_image // 2
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)

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
        visual.draw_caption(im, (10, 30), display_text)

        # Output video file
        if video_writer != None:
            video_writer.write(im)

        # Display
        cv2.imshow(WINDOW_NAME, im)
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

                
    # When everything done, release the window
    cap.release()
    if video_writer != None:
        video_writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
