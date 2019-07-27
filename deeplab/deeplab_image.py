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
import time
import os


import numpy as np
from edgetpu.basic.basic_engine import BasicEngine
from PIL import Image

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


def create_pascal_label_colormap():
    """Creates a label colormap used in PASCAL VOC segmentation benchmark.

  Returns:
    A Colormap for visualizing segmentation results.
  """
    colormap = np.zeros((256, 3), dtype=np.uint8)
    ind = np.arange(256, dtype=np.uint8)

    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:, channel] |= ((ind >> channel) & 1) << shift
        ind >>= 3

    return colormap


def label_to_color_image(label):
    """Adds color defined by the dataset colormap to the label.

  Args:
    label: A 2D array with integer type, storing the segmentation label.

  Returns:
    result: A 2D array with floating type. The element of the array
      is the color indexed by the corresponding element in the input label
      to the PASCAL color map.

  Raises:
    ValueError: If label is not of rank 2 or its value is larger than color
      map maximum entry.
  """
    if label.ndim != 2:
        raise ValueError("Expect 2-D input label")

    colormap = create_pascal_label_colormap()

    if np.max(label) >= len(colormap):
        raise ValueError("label value too large.")

    return colormap[label]


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
    FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP)

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

    seg_image = label_to_color_image(seg_map)
    seg_image = Image.fromarray(seg_image).resize((im_width, im_height), Image.NEAREST)
    out_image = np.array(org_img) * 0.5 + np.array(seg_image) * 0.5

    pil_img = Image.fromarray(out_image.astype(np.uint8))
    pil_img.save(os.path.join(".", "save.png"))


if __name__ == "__main__":
    main()

