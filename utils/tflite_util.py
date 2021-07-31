#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    TensorFlow Lite Utils.

    Copyright (c) 2020 Nobuo Tsukamoto

    This software is released under the MIT License.
    See the LICENSE file in the project root for more information.
"""

import os

from ctypes import *
import numpy as np
import tflite_runtime.interpreter as tflite
import platform

EDGETPU_SHARED_LIB = {
    "Linux": "libedgetpu.so.1",
    "Darwin": "libedgetpu.1.dylib",
    "Windows": "edgetpu.dll",
}[platform.system()]


def make_interpreter(model_file, num_of_threads, delegate_library=None):
    """ make tf-lite interpreter.

    Args:
        model_file: Model file path.
        num_of_threads: Num of threads.
        delegate_library: Delegate file path.

    Return:
        tf-lite interpreter.
    """

    if "edgetpu.tflite" in model_file:
        print("EdgeTpu delegate")
        return tflite.Interpreter(
            model_path=model_file,
            experimental_delegates=[
                tflite.load_delegate(EDGETPU_SHARED_LIB)
            ],
        )
    elif delegate_library is not None:

        print("{} delegate".format(os.path.splitext(os.path.basename(delegate_library))[0]))
        option = {"backends": "CpuAcc",
                  "logging-severity": "info",
                  "number-of-threads": str(num_of_threads),
                  "enable-fast-math":"true"}
        print(option)
        return tflite.Interpreter(
            model_path=model_file,
            experimental_delegates=[
                tflite.load_delegate(delegate_library, options=option)
            ],
        )
    else:
        return tflite.Interpreter(model_path=model_file, num_threads=num_of_threads)


def set_input_tensor(interpreter, image):
    """ Sets the input tensor.

    Args:
        interpreter: Interpreter object.
        image: a function that takes a (width, height) tuple, 
        and returns an RGB image resized to those dimensions.
    """
    tensor_index = interpreter.get_input_details()[0]["index"]
    input_tensor = interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = image.copy()


def get_output_tensor(interpreter, index):
    """ Returns the output tensor at the given index.

    Args:
        interpreter
        index

    Returns:
        tensor
    """
    output_details = interpreter.get_output_details()[index]
    tensor = np.squeeze(interpreter.get_tensor(output_details["index"]))
    return tensor
