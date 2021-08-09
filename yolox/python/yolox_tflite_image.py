#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    TensorFlow Lite YOLOX with OpenCV.

    Copyright (c) 2021 Nobuo Tsukamoto

    This software is released under the MIT License.
    See the LICENSE file in the project root for more information.
"""

import argparse
import os
import random
import time

import cv2
import numpy as np
from utils import visualization as visual

from utils.label_util import read_label_file
from utils.tflite_util import (get_output_tensor, make_interpreter,
                               set_input_tensor)

from yolox.utils.demo_utils import multiclass_nms, demo_postprocess

COCO_CLASSES = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)


# def preprocess(image, input_size, mean, std, swap=(2, 0, 1)):
def preprocess(image, input_size, mean, std):
    # https://github.com/Megvii-BaseDetection/YOLOX/blob/c4714bb97c2f13d26195544d5f9e1ea91241ee2b/yolox/data/data_augment.py#L165 noqa: E501

    if len(image.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3)) * 114.0
    else:
        padded_img = np.ones(input_size) * 114.0
    img = np.array(image)
    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR
    ).astype(np.float32)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img
    image = padded_img

    image = image.astype(np.float32)
    image = image[:, :, ::-1]
    image /= 255.0
    if mean is not None:
        image -= mean
    if std is not None:
        image /= std
    # image = image.transpose((2, 0, 1))
    image = np.ascontiguousarray(image, dtype=np.float32)
    return image, r


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="File path of Tflite model.", required=True)
    parser.add_argument("--label", help="File path of label file.", required=True)
    parser.add_argument(
        "--threshold", help="threshold to filter results.", default=0.5, type=float
    )
    parser.add_argument("--thread", help="Num threads.", default=2, type=int)
    parser.add_argument("--input", help="File path of Videofile.", default="")
    parser.add_argument("--output", help="File path of result.", default="")
    parser.add_argument(
        "--with_p6",
        action="store_true",
        help="Whether your model uses p6 in FPN/PAN.",
    )
    args = parser.parse_args()

    # Initialize TF-Lite interpreter.
    interpreter = make_interpreter(args.model, args.thread)
    interpreter.allocate_tensors()
    _, height, width, channel = interpreter.get_input_details()[0]["shape"]
    input_shape = (320, 320)
    print("Interpreter(height, width, channel): ", height, width, channel)

    # Read label and generate random colors.
    labels = read_label_file(args.label) if args.label else None
    last_key = sorted(labels.keys())[len(labels.keys()) - 1]
    random.seed(42)
    colors = visual.random_colors(last_key)

    input_im = cv2.imread(args.input)
    im, ratio = preprocess(input_im, input_shape, mean, std)

    # Run inference.
    set_input_tensor(interpreter, im[None, :, :, :])
    interpreter.invoke()
    output_details = interpreter.get_output_details()[0]
    output = interpreter.get_tensor(output_details["index"])

    print(output.shape)
    print(output)
    output = output.transpose((0, 2, 1))

    predictions = demo_postprocess(output, input_shape, p6=args.with_p6)[0]

    boxes = predictions[:, :4]
    scores = predictions[:, 4:5] * predictions[:, 5:]

    boxes_xyxy = np.ones_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.
    boxes_xyxy /= ratio
    # print(boxes, scores)
    dets = multiclass_nms(boxes_xyxy, scores, nms_thr=0.65, score_thr=0.1)

    print(dets)
    # Display result.
    if dets is not None:
        final_boxes = dets[:, :4]
        final_scores = dets[:, 4]
        final_cls_inds = dets[:, 5]

        for i, box in enumerate(final_boxes):
            class_id = int(final_cls_inds[i])
            score = final_scores[i]
            if score < args.threshold:
                print("a")
                continue

            xmin = int(box[0])
            ymin = int(box[1])
            xmax = int(box[2])
            ymax = int(box[3])
            print(xmin, ymin, xmin, ymax)
            caption = "{0}({1:.2f})".format(labels[class_id], score)

            # Draw a rectangle and caption.
            visual.draw_rectangle(input_im, (xmin, ymin, xmax, ymax), colors[class_id])
            visual.draw_caption(input_im, (xmin, ymin, xmax, ymax), caption)

    cv2.imwrite(args.output, input_im)


if __name__ == "__main__":
    main()
