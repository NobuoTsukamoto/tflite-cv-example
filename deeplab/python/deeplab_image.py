#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Edge TPU image segmentation run single image.

    Copyright (c) 2019 Nobuo Tsukamoto

    This software is released under the MIT License.
    See the LICENSE file in the project root for more information.
"""

import argparse
import io
import os
import time

import numpy as np
from PIL import Image

from edgetpu.basic.basic_engine import BasicEngine
from utils import label_util

LABEL_NAMES = np.asarray(
    [
        "background",
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "sofa",
        "train",
        "tv",
    ]
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="File path of Tflite model.", required=True)
    parser.add_argument(
        "--image", help="File path of the image to be recognized.", required=True
    )
    # parser.add_argument(
    #    '--label', help='File path of label file.', required=True)
    args = parser.parse_args()

    FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
    FULL_COLOR_MAP = label_util.label_to_color_image(FULL_LABEL_MAP)

    # Read image
    org_img = Image.open(args.image)
    im_width, im_height = org_img.size
    engine = BasicEngine(args.model)

    _, height, width, _ = engine.get_input_tensor_shape()
    img = org_img.resize((width, height), Image.NEAREST)
    input_tensor = np.asarray(img).flatten()

    latency, result = engine.RunInference(input_tensor)

    seg_map = np.array(result, dtype=np.uint8)
    seg_map = np.reshape(seg_map, (width, height))

    seg_image = label_util.label_to_color_image(seg_map)
    seg_image = Image.fromarray(seg_image).resize((im_width, im_height), Image.NEAREST)
    out_image = np.array(org_img) * 0.5 + np.array(seg_image) * 0.5

    pil_img = Image.fromarray(out_image.astype(np.uint8))
    pil_img.save(os.path.join(".", "save.png"))


if __name__ == "__main__":
    main()
