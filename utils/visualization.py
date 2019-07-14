#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Copyright (c) 2019 Nobuo Tsukamoto

    Visualizetion functions.

    This software is released under the MIT License.
    See the LICENSE file in the project root for more information.
"""
import colorsys
import random

import numpy as np
import cv2

def random_colors(N):
    """ Random color generator.
    """
    N = N + 1
    hsv = [(i / N, 1.0, 1.0) for i in range(N)]
    colors = list(map(lambda c: tuple(int(i * 255) for i in colorsys.hsv_to_rgb(*c)), hsv))
    random.shuffle(colors)
    return colors

def draw_rectangle(image, box, color, thickness=3):
    """ Draws a rectangle.

    Args:
        image: The image to draw on.
        box: A list of 4 elements (x1, y1, x2, y2).
        color: Rectangle color.
        thickness: Thickness of lines.
    """
    b = np.array(box).astype(int)
    cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), color, thickness)

def draw_caption(image, box, caption):
    """ Draws a caption above the box in an image.

    Args:
        image: The image to draw on.
        box: A list of 4 elements (x1, y1, x2, y2).
        caption: String containing the text to draw.
    """
    b = np.array(box).astype(int)
    cv2.putText(image, caption, (b[0], b[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(image, caption, (b[0], b[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)



