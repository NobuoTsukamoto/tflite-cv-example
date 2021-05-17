#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    TensorFlow Lite Object detection coco benchmark.

    Copyright (c) 2021 Nobuo Tsukamoto

    This software is released under the MIT License.
    See the LICENSE file in the project root for more information.
"""

import os
import argparse
import time
import json

import cv2
from utils.tflite_util import get_output_tensor, make_interpreter, set_input_tensor

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def get_minival_ids(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()

    ret = []
    for line in lines:
        ret.append(int(line))
    return ret


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="File path of Tflite model.", required=True)
    parser.add_argument("--thread", help="Num threads.", default=2, type=int)
    parser.add_argument(
        "--images_dir",
        type=str,
        default=None,
        help="Directory containing validation set TFRecord files.",
    )
    parser.add_argument(
        "--annotation_path", type=str, help="Path that contains COCO annotations"
    )
    parser.add_argument(
        "--allowlist_file",
        type=str,
        help="File with COCO image ids to preprocess, one on each line.",
        default=None,
        required=False,
    )
    args = parser.parse_args()
    parser.add_argument(
        "--display_every",
        type=int,
        default=100,
        help="Number of iterations executed between two consecutive display of metrics",
    )
    parser.add_argument(
        "--num_warmup_iterations",
        type=int,
        default=50,
        help="Number of initial iterations skipped from timing",
    )
    parser.add_argument(
        "--minival_ids", type=str, help="Path that minival ids list file.", default=None
    )
    parser.add_argument(
        "--exec_coco_metrics", help="Execute coco metrics.", action='store_true'
    )
    args = parser.parse_args()

    # Initialize TF-Lite interpreter.
    interpreter = make_interpreter(args.model, args.thread)
    interpreter.allocate_tensors()
    batch, height, width, channel = interpreter.get_input_details()[0]["shape"]
    print("Interpreter: ", batch, height, width, channel)

    # COCO Datasets.
    coco = COCO(annotation_file=args.annotation_path)

    # Image IDs
    if args.allowlist_file is None:
        image_ids = coco.getImgIds()
    else:
        with open(args.allowlist_file, "r") as allowlist:
            image_ids = set([int(x) for x in allowlist.readlines()])

    num_steps = len(image_ids)
    elapsed_list = []
    coco_detections = []

    for i, image_id in enumerate(image_ids):
        coco_img = coco.imgs[image_id]
        image_width = coco_img["width"]
        image_height = coco_img["height"]

        # Load image.
        im = cv2.imread(os.path.join(args.images_dir, coco_img["file_name"]))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        resize_im = cv2.resize(im, (width, height))

        # Run inference.
        start = time.perf_counter()

        set_input_tensor(interpreter, resize_im)
        interpreter.invoke()
        boxes = get_output_tensor(interpreter, 0)
        class_ids = get_output_tensor(interpreter, 1)
        scores = get_output_tensor(interpreter, 2)
        count = int(get_output_tensor(interpreter, 3))

        inference_time = time.perf_counter() - start
        elapsed_list.append(inference_time)

        for j in range(count):
            y1, x1, y2, x2 = boxes[j]
            bbox_coco_fmt = [
                x1 * image_width,  # x0
                y1 * image_height,  # x1
                (x2 - x1) * image_width,  # width
                (y2 - y1) * image_height,  # height
            ]
            coco_detection = {
                "image_id": image_id,
                "category_id": int(class_ids[j]) + 1,
                "bbox": [int(coord) for coord in bbox_coco_fmt],
                "score": float(scores[j]),
            }
            coco_detections.append(coco_detection)

        if (i + 1) % args.display_every == 0:
            print(
                "  step %03d/%03d, iter_time(ms)=%.0f"
                % (i + 1, num_steps, elapsed_list[-1] * 1000)
            )

    # write coco detections to file
    coco_detections_path = os.path.join(".", "coco_detections.json")
    with open(coco_detections_path, "w") as f:
        json.dump(coco_detections, f)

    cocoDt = coco.loadRes(coco_detections_path)

    if args.exec_coco_metrics is not None:
        # compute coco metrics
        eval = COCOeval(coco, cocoDt, "bbox")
        eval.params.imgIds = image_ids

        eval.evaluate()
        eval.accumulate()
        eval.summarize()

        os.remove(coco_detections_path)


if __name__ == "__main__":
    main()
