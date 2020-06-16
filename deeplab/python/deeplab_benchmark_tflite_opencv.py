#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Edge TPU DeepLabv3 Image segmenation benchmark with OpenCV.

    Copyright (c) 2020 Nobuo Tsukamoto

    This software is released under the MIT License.
    See the LICENSE file in the project root for more information.
"""
import argparse
import io
import os
import time

import numpy as np

import tflite_runtime.interpreter as tflite
import platform

import cv2

def make_interpreter(model_file, num_of_threads):
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
        return tflite.Interpreter(model_path=model_file, num_threads=num_of_threads)


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
    parser.add_argument('--image', help='File path of image file.', required=True)
    parser.add_argument('--count', help='Repeat count.', default=100, type=int)
    parser.add_argument("--thread", help="Num threads.", default=2, type=int)
    args = parser.parse_args()

    # Initialize TF-Lite interpreter.
    interpreter = make_interpreter(args.model, args.thread)
    interpreter.allocate_tensors()
    _, height, width, channel = interpreter.get_input_details()[0]['shape']
    print('Interpreter: ', height, width, channel)

    # Load image.
    im = cv2.imread(args.image)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    resize_im = cv2.resize(im, (width, height))

    model_file, *device = args.model.split('@')
    model_name = os.path.splitext(os.path.basename(model_file))[0]

    elapsed_list = []
    
    for i in range(args.count + 1):
        
        # Run inference.
        start = time.perf_counter()

        set_input_tensor(interpreter, resize_im)
        interpreter.invoke()
        seg_map = get_output(interpreter)

        inference_time = (time.perf_counter() - start) * 1000
        if i is 0:
            print('First Inference : {0:.2f} ms'.format(inference_time))
        else:
            elapsed_list.append(inference_time)

    print('Inference : {0:.2f} ms'.format(np.array(elapsed_list).mean()))


if __name__ == "__main__":
    main()
